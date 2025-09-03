nextflow.enable.dsl=2

params.mode="preprocess"
params.dataDir="/aloy/home/ymartins/match_clinical_trial/validation_trials"

params.help = false
if (params.help) {
    log.info params.help_message
    exit 0
}

log.info """\
 PICO Augmentation workflow  -  P I P E L I N E
 ===================================
 dataDir       : ${params.dataDir}
 runningConfig : ${params.runningConfig}
 mode       : ${params.mode}
 """

process setEnvironment {
    
    output:
    val 1, emit: flag

    script:
        """
        if [ ! -d "${params.dataDir}" ]; then
            mkdir ${params.dataDir}
        fi
        """
}

process PROCESS_PreprocessCT {
    input:
    path outDir 
    path parameterFile
    val flow
    
    output:
    path "$outDir/logs/tasks_process_ct.ready"
        
    script:
        "python3 ${projectDir}/modules/process_ct.py -execDir $outDir -paramFile $parameterFile "
}

process PROCESS_PreprocessPubmed {
    input:
    path outDir 
    path parameterFile
    val flow
    
    output:
    path "$outDir/logs/tasks_process_pubmed.ready"
        
    script:
        "python3 ${projectDir}/modules/process_pubmed.py -execDir $outDir -paramFile $parameterFile "
}

process PROCESS_Validation {
    input:
    path outDir 
    path parameterFile
    val flow
    
    output:
    path "$outDir/logs/tasks_validation.ready"
        
    script:
        "python3 ${projectDir}/modules/validation.py -execDir $outDir -paramFile $parameterFile "
}

workflow {
    result = setEnvironment()

    if( params.mode == "preprocess" | params.mode == "all" ){
        result = PROCESS_PreprocessCT( params.dataDir, params.runningConfig, result )
        result = PROCESS_PreprocessPubmed( params.dataDir, params.runningConfig, result )
    }

    if( params.mode == "validation" | params.mode == "all" ){
        result = PROCESS_Validation( params.dataDir, params.runningConfig, result )
    }

}

// nextflow run -bg /aloy/home/ymartins/match_clinical_trial/ctpico_validation_workflow/main.nf --dataDir /aloy/home/ymartins/match_clinical_trial/validation_trials/ --runningConfig /aloy/home/ymartins/match_clinical_trial/ctpico_validation_workflow/validation_config.json --mode "all"

workflow.onComplete {
    log.info ( workflow.success ? "\nDone! Models were trained and evaluated!\n" : "Oops .. something went wrong" )
}
