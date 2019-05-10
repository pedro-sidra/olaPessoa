# Olá Pessoa

Script utilizando a biblioteca "face\_recognition" para anunciar o nome de
pessoas que passam pelo campo de visão de uma camera.

## Funcionamento
Passo-a-passo:
* Lê imagens da pasta "Face", que devem conter a face de uma pessoa. O nome da imagem é usado como o nome da pessoa
* Salva imagens de pessoas não-reconhecidas em um buffer na pasta do script. 
    * Para adicionar uma pessoa, basta renomear uma das imagens de desconhecidos com o nome correspondente e botar na pasta Face
* Quando reconhece alguém na imagem, anuncia o nome em voz alta. A mensagem pode ser personalizada para cada pessoa, editando o arquivo greetings.txt (que é gerado após a primeira execução)

## Melhorias
Lista de melhorias pendentes:

* Refatorar o código para usar classes e melhorar a manutenção
* Utilizar argumentos na linha de comando no arquivo recognize\_video
