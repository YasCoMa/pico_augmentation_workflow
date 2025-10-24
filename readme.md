# CTPICO wf - Workflow to perform datasets augmentation for PICO NER experiments

Nextflow workflow to perform automatic dataset augmention of a specific set of named entities named PICO (standing for participants, intervention, control and outcomes), gathering and cross-referencing data from ClinicalTrials (CTs) ncbi API and Pubmed abstracts. 

<div style="text-align: center">
	<img src="pipeline.png" alt="pipeline"
	title="CTPICO workflow" width="550px" />
</div>

## Summary

We have developed a workflow with two main modules: (1.1) Acquire/Process up to date completed CTs raw data and validate the presence of the main information required for the entities validation; (1.2) Gather/process pubmed abstracts directly linked  in the raw CT structured files; (2) Validation by confronting and testing pairwise similarity between the prediction annotations per PICO-domain entity from NER Fair workflow output and the items extracted for each entity in the clinical trial associated with the respective pubmed ID. At the end, it generates the table with the similarity scores for the annotations and also a folder containing \*.ann and \*.txt ready to serve as input to the NER Fair workflow for training a new model.

## Requirements:
* The packages are stated in the environment's exported file: environment.yml

## Usage Instructions
### Preparation:
1. ````git clone https://github.com/YasCoMa/pico_augmentation_workflow.git````
2. ````cdpico_augmentation_workflow````
3. ````conda env create --file environment.yml````
4. The workflow requires four parameters, you can edit them in main.nf, or pass in the command line when you start the execution. The parameters are:
	- **mode**: Indicates the goal of the workflow: 'preprocess' or 'validation'. It activates accordingly the steps according to the mode.
	- **dataDir**: The directory where the generated files will be stored
	- **runningConfig**: A json file with the configuration setup desired by the user. Each main key of the json file is explained below.
		- datasets: The datasets available for analysis - Example: ``["hla", "bcipep", "gram+_epitope", "gram-_epitope", "gram+_protein", "gram-_protein", "allgram_epitope", "allgram_protein"]``


### Run workflow:
1. Examples of running configuration are shown in running_config.json and eskape_running_config.json

2. Modes of execution:
	- **Run Pre-processing:**
		- ````nextflow run main.nf --dataDir /path/to/paprec_data --runningConfig /path/to/running_config.json --mode preprocess ````
	- **Run Validation:**
		- ````nextflow run main.nf --dataDir /path/to/paprec_data --runningConfig /path/to/running_config.json --mode validation ````

## Reference

## Bug Report
Please, use the [Issues](https://github.com/YasCoMa/pico_augmentation_workflow/issues) tab to report any bug.