"""Precision, recall e F1 do e2e contra o groundtruth do ChokePoint.

Duas famílias de métrica, calculadas sobre a mesma passagem do vídeo:

**Detecção.** O groundtruth não traz caixa, só as coordenadas dos dois olhos do
sujeito anotado. O casamento é, então, "os dois olhos caem dentro da caixa
detectada", com margem relativa ao tamanho da caixa. Cada pessoa anotada consome
no máximo uma detecção, escolhida pela maior confiança; detecção que não casa com
ninguém é falso positivo.

**Reconhecimento.** Só sobre as detecções que casaram, porque só aí existe
identidade de referência. É classificação multiclasse sobre os IDs da galeria mais
o rótulo `DESCONHECIDO`, para quando nenhum candidato passa do limiar operacional.

Aviso registrado de propósito: a anotação do ChokePoint marca **um** sujeito de
interesse por quadro, não toda face visível. Em `P1E_S1_C1` há 88 quadros com uma
segunda pessoa real não anotada e 262 quadros dados como vazios que têm rosto
detectável. Logo o **recall** desta métrica é fiel, mas a **precision** mede
também a lacuna da anotação, não só erro do detector. Ver docs/DATASET_CHOKEPOINT.md.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

DESCONHECIDO = "DESCONHECIDO"
MARGEM_PADRAO = 0.15


@dataclass(frozen=True)
class PessoaAnotada:
    identidade: str
    olho_esq: tuple[float, float]
    olho_dir: tuple[float, float]

    @property
    def iod(self) -> float:
        dx = self.olho_esq[0] - self.olho_dir[0]
        dy = self.olho_esq[1] - self.olho_dir[1]
        return (dx * dx + dy * dy) ** 0.5


def normalizar_id(bruto: Optional[str]) -> Optional[str]:
    """`0003`, `ID0003` e `id_3` viram `ID0003`.

    A anotação usa quatro dígitos crus; a galeria, o nome do arquivo (`ID0003`).
    Sem normalizar, toda comparação de identidade falharia silenciosamente.
    """
    if bruto is None:
        return None
    digitos = re.sub(r"\D", "", str(bruto))
    if not digitos:
        return None
    return "ID%04d" % int(digitos)


def carregar_groundtruth(caminho: str | Path) -> list[list[PessoaAnotada]]:
    """Uma entrada por `<frame>`, na ordem do arquivo.

    A ordem importa: o i-ésimo `<frame>` corresponde ao i-ésimo JPEG e, portanto,
    ao i-ésimo quadro do MP4 montado por scripts/frames_to_video.py. A numeração
    do atributo `number` é esparsa (0 a 4409 para 2.292 quadros) e não serve de
    índice.
    """
    raiz = ET.parse(str(caminho)).getroot()
    por_quadro: list[list[PessoaAnotada]] = []
    for quadro in raiz.findall("frame"):
        pessoas = []
        for pessoa in quadro.findall("person"):
            esq, dir_ = pessoa.find("leftEye"), pessoa.find("rightEye")
            if esq is None or dir_ is None:
                # Existem dois casos assim no dataset inteiro, ambos em
                # P2E_S3_C1.1. Sem os dois olhos nao ha como casar por olho.
                continue
            ident = normalizar_id(pessoa.get("id"))
            if ident is None:
                continue
            pessoas.append(
                PessoaAnotada(
                    identidade=ident,
                    olho_esq=(float(esq.get("x")), float(esq.get("y"))),
                    olho_dir=(float(dir_.get("x")), float(dir_.get("y"))),
                )
            )
        por_quadro.append(pessoas)
    return por_quadro


def _dentro(caixa: Iterable[float], ponto: tuple[float, float], margem: float) -> bool:
    x1, y1, x2, y2 = (float(v) for v in caixa)
    largura, altura = x2 - x1, y2 - y1
    return (
        x1 - margem * largura <= ponto[0] <= x2 + margem * largura
        and y1 - margem * altura <= ponto[1] <= y2 + margem * altura
    )


@dataclass
class Contagens:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        d = self.tp + self.fp
        return self.tp / d if d else 0.0

    @property
    def recall(self) -> float:
        d = self.tp + self.fn
        return self.tp / d if d else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def como_dict(self) -> dict[str, Any]:
        return {
            "tp": self.tp, "fp": self.fp, "fn": self.fn,
            "precision": round(self.precision, 6),
            "recall": round(self.recall, 6),
            "f1": round(self.f1, 6),
        }


@dataclass
class Avaliador:
    """Acumula quadro a quadro e fecha as métricas no fim."""

    groundtruth: list[list[PessoaAnotada]]
    margem: float = MARGEM_PADRAO
    deteccao: Contagens = field(default_factory=Contagens)
    # Duas leituras do reconhecimento, porque respondem a perguntas diferentes:
    #   e2e  -- face anotada que o detector perdeu conta como erro de reconhecimento,
    #           que e o numero honesto do sistema inteiro;
    #   dado -- so as faces que o detector achou, que isola o extrator e a galeria.
    reconhecimento: dict[str, Contagens] = field(default_factory=dict)
    rec_dado_deteccao: dict[str, Contagens] = field(default_factory=dict)
    rank1_acertos: int = 0
    rank1_total: int = 0
    abaixo_do_limiar: int = 0
    quadros_vistos: int = 0
    _iod_perdidas: list[float] = field(default_factory=list)

    def _classe(self, nome: str) -> Contagens:
        return self.reconhecimento.setdefault(nome, Contagens())

    def _classe_dado(self, nome: str) -> Contagens:
        return self.rec_dado_deteccao.setdefault(nome, Contagens())

    def adicionar_quadro(self, indice: int, reconhecimentos: list[dict[str, Any]]) -> None:
        """`reconhecimentos` é o que FaceRecognizer.recognize_frame devolveu."""
        if indice >= len(self.groundtruth):
            raise IndexError(
                f"quadro {indice} além do groundtruth ({len(self.groundtruth)} quadros). "
                "Rode com --no-loop: com loop o índice do quadro deixa de casar com a anotação."
            )
        self.quadros_vistos += 1
        pessoas = self.groundtruth[indice]

        # Maior confiança primeiro: quando duas caixas contêm os mesmos olhos, a
        # melhor vira TP e a outra, FP -- e nao duas TP para uma pessoa.
        candidatas = sorted(
            enumerate(reconhecimentos),
            key=lambda par: float(par[1].get("detection_score") or 0.0),
            reverse=True,
        )
        usadas: set[int] = set()

        for pessoa in pessoas:
            escolhida = None
            for idx, rec in candidatas:
                if idx in usadas:
                    continue
                caixa = rec.get("bbox")
                if caixa is None:
                    continue
                if _dentro(caixa, pessoa.olho_esq, self.margem) and _dentro(
                    caixa, pessoa.olho_dir, self.margem
                ):
                    escolhida = (idx, rec)
                    break
            if escolhida is None:
                self.deteccao.fn += 1
                self._classe(pessoa.identidade).fn += 1
                self._iod_perdidas.append(pessoa.iod)
                continue

            idx, rec = escolhida
            usadas.add(idx)
            self.deteccao.tp += 1

            verdadeiro = pessoa.identidade
            previsto = normalizar_id(rec.get("entity_id")) if rec.get("recognized") else None
            if previsto is None:
                previsto = DESCONHECIDO
                self.abaixo_do_limiar += 1

            if previsto == verdadeiro:
                self._classe(verdadeiro).tp += 1
                self._classe_dado(verdadeiro).tp += 1
            else:
                self._classe(verdadeiro).fn += 1
                self._classe(previsto).fp += 1
                self._classe_dado(verdadeiro).fn += 1
                self._classe_dado(previsto).fp += 1

            # Rank-1 ignora o limiar: mede o extrator, nao a politica de corte.
            topo = normalizar_id(rec.get("top_candidate_entity_id"))
            self.rank1_total += 1
            if topo == verdadeiro:
                self.rank1_acertos += 1

        self.deteccao.fp += len(reconhecimentos) - len(usadas)

    def resumo(self) -> dict[str, Any]:
        def agregar(por_classe: dict[str, Contagens]) -> dict[str, Any]:
            classes = sorted(
                n for n, c in por_classe.items()
                if n != DESCONHECIDO and (c.tp + c.fn) > 0
            )
            # Cada caso tem um rotulo verdadeiro e um previsto, entao todo erro e
            # ao mesmo tempo FN da classe certa e FP da prevista: no micro,
            # FP == FN e micro-P = micro-R = micro-F1 = acuracia.
            micro = Contagens()
            for nome in classes:
                micro.tp += por_classe[nome].tp
                micro.fn += por_classe[nome].fn
            micro.fp = micro.fn

            def media(campo: str) -> float:
                if not classes:
                    return 0.0
                return sum(getattr(por_classe[n], campo) for n in classes) / len(classes)

            return {
                "identidades_na_referencia": len(classes),
                "casos": micro.tp + micro.fn,
                "acertos": micro.tp,
                "macro_precision": round(media("precision"), 6),
                "macro_recall": round(media("recall"), 6),
                "macro_f1": round(media("f1"), 6),
                "micro_precision": round(micro.precision, 6),
                "micro_recall": round(micro.recall, 6),
                "micro_f1": round(micro.f1, 6),
                "por_identidade": {n: por_classe[n].como_dict() for n in classes},
            }

        iod = sorted(self._iod_perdidas)
        anotadas = sum(len(p) for p in self.groundtruth[: self.quadros_vistos])
        return {
            "quadros_avaliados": self.quadros_vistos,
            "quadros_no_groundtruth": len(self.groundtruth),
            "pessoas_anotadas": anotadas,
            "margem_casamento": self.margem,
            "deteccao": self.deteccao.como_dict(),
            "reconhecimento_e2e": agregar(self.reconhecimento),
            "reconhecimento_dado_deteccao": agregar(self.rec_dado_deteccao),
            "rank1": {
                "dado_deteccao": round(self.rank1_acertos / self.rank1_total, 6) if self.rank1_total else 0.0,
                "e2e": round(self.rank1_acertos / anotadas, 6) if anotadas else 0.0,
                "acertos": self.rank1_acertos,
                "casadas": self.rank1_total,
                "anotadas": anotadas,
            },
            "previstas_desconhecido": self.abaixo_do_limiar,
            "faces_perdidas_iod": {
                "n": len(iod),
                "min": round(iod[0], 2) if iod else None,
                "mediana": round(iod[len(iod) // 2], 2) if iod else None,
                "max": round(iod[-1], 2) if iod else None,
            },
        }


__all__ = [
    "Avaliador",
    "Contagens",
    "PessoaAnotada",
    "carregar_groundtruth",
    "normalizar_id",
    "DESCONHECIDO",
    "MARGEM_PADRAO",
]
