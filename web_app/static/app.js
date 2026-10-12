const form =
    document.getElementById("personForm");

const nameInput =
    document.getElementById("name");

const filesInput =
    document.getElementById("files");

const submitButton =
    document.getElementById("submitButton");

const statusBox =
    document.getElementById("status");

const preview =
    document.getElementById("preview");


filesInput.addEventListener(
    "change",
    function () {
        preview.innerHTML = "";

        const files =
            Array.from(filesInput.files);

        files.forEach((file) => {

            const image =
                document.createElement("img");

            image.src =
                URL.createObjectURL(file);

            preview.appendChild(image);
        });
    }
);


form.addEventListener(
    "submit",
    async function (event) {

        event.preventDefault();

        const name =
            nameInput.value.trim();

        const files =
            Array.from(filesInput.files);

        if (!name) {
            showError(
                "Informe o nome da pessoa."
            );

            return;
        }

        if (files.length === 0) {
            showError(
                "Selecione pelo menos uma foto."
            );

            return;
        }

        const formData =
            new FormData();

        formData.append(
            "name",
            name
        );

        files.forEach((file) => {
            formData.append(
                "files",
                file
            );
        });


        submitButton.disabled = true;

        submitButton.textContent =
            "Cadastrando...";

        statusBox.className =
            "status";

        statusBox.textContent =
            "Processando imagens...";


        try {

            const response =
                await fetch(
                    "/persons",
                    {
                        method: "POST",
                        body: formData
                    }
                );


            const data =
                await response.json();


            if (!response.ok) {
                throw new Error(
                    JSON.stringify(
                        data.detail
                    )
                );
            }


            showSuccess(
                "Pessoa cadastrada com sucesso!\n\n" +
                "Nome: " +
                data.name +
                "\n" +
                "Identificador: " +
                data.entity_id +
                "\n" +
                "Fotos cadastradas: " +
                data.images_registered +
                "\n" +
                "Erros: " +
                data.errors_count
            );


            form.reset();

            preview.innerHTML = "";

        }
        catch (error) {

            showError(
                "Erro ao cadastrar pessoa:\n" +
                error.message
            );

        }
        finally {

            submitButton.disabled =
                false;

            submitButton.textContent =
                "Cadastrar pessoa";
        }
    }
);


function showSuccess(message) {

    statusBox.className =
        "status success";

    statusBox.textContent =
        message;
}


function showError(message) {

    statusBox.className =
        "status error";

    statusBox.textContent =
        message;
}
