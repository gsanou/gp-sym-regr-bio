# gp-sym-regr-bio

Evaluation of genetic programming symbolic regression vs standard supervised machine learning methods for prediction in biological data case studies.

## Installation

Install and set up [Miniconda3](https://docs.conda.io/en/latest/miniconda.html)

Set up repository and conda environment
```bash
git clone git@github.com:hermidalc/gplearn-vs-sklearn.git
cd gplearn-vs-sklearn
conda env create -f environment.yml
pip install gplearn
```

## Data processing

Download TCGA BRCA RNA-seq normalized FPKM data for ductal and lobular carcinoma primary tumors from NCI GDC. From the facet query add the 985 files to the cart and download the cart, clinical and sample sheet.  The URL below is the GDC facet query for:

`
cases.diagnoses.primary_diagnosis in ["infiltrating duct carcinoma, nos","lobular carcinoma, nos"] and cases.project.project_id in ["TCGA-BRCA"] and cases.samples.sample_type in ["primary tumor"] and files.analysis.workflow_type in ["HTSeq - FPKM"]
`

https://portal.gdc.cancer.gov/repository?facetTab=files&filters=%7B%22op%22%3A%22and%22%2C%22content%22%3A%5B%7B%22op%22%3A%22in%22%2C%22content%22%3A%7B%22field%22%3A%22cases.diagnoses.primary_diagnosis%22%2C%22value%22%3A%5B%22infiltrating%20duct%20carcinoma%2C%20nos%22%2C%22lobular%20carcinoma%2C%20nos%22%5D%7D%7D%2C%7B%22op%22%3A%22in%22%2C%22content%22%3A%7B%22field%22%3A%22cases.project.project_id%22%2C%22value%22%3A%5B%22TCGA-BRCA%22%5D%7D%7D%2C%7B%22op%22%3A%22in%22%2C%22content%22%3A%7B%22field%22%3A%22cases.samples.sample_type%22%2C%22value%22%3A%5B%22primary%20tumor%22%5D%7D%7D%2C%7B%22op%22%3A%22in%22%2C%22content%22%3A%7B%22field%22%3A%22files.analysis.workflow_type%22%2C%22value%22%3A%5B%22HTSeq%20-%20FPKM%22%5D%7D%7D%5D%7D

Unpack the download tar.gz file and sample sheet to a directory and run the following script to generate the data matrix:

```bash
./create_data_matrix.pl --data-dir <path to directory>
```
