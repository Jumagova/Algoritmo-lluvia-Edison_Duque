FROM continuumio/anaconda3:latest

COPY lluvia_plateromodEdison.py /lluvia_plateromodEdison.py 

RUN conda create --name algoritmolluvia python=3.10.6 && \
    echo "conda activate algoritmolluvia" >> ~/.bashrc

SHELL ["/bin/bash", "--login", "-c"]
RUN source ~/.bashrc && \
    pip install soundfile && \
    conda install pandas && \
    conda install numpy && \
    pip install scipy && \
    conda install -c conda-forge tqdm && \
    pip install openpyxl

CMD  python lluvia_plateromodEdison.py -Edison_Duque False -p 'grabaciones' -pr 'resultados' -name 'ResultGrab.xlsx' -seg 60