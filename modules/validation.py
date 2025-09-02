import os
import re
import sys
import json
import glob
import pickle
import Levenshtein
import numpy as np
import pandas as pd
from tqdm import tqdm

import logging
import argparse

class ExperimentValidationBySimilarity:
    def __init__(self):
        
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(filename='log_validation.log', level=logging.INFO)
        
        self._get_arguments()
        self._setup_out_folders()

        self.fready = os.path.join( self.logdir, "tasks_validation.ready")
        self.ready = os.path.exists( self.fready )
        if(self.ready):
            self.logger.info("----------- Validation step skipped since it was already computed -----------")
            self.logger.info("----------- Validation step ended -----------")
        
    def __setup_logdir(self, execdir):
        self.logdir = os.path.join( execdir, "logs" )
        if( not os.path.exists(self.logdir) ):
            os.makedirs( self.logdir )

        logf = os.path.join( self.logdir, "validation.log" )
        logging.basicConfig( filename=logf, encoding="utf-8", filemode="a", level=logging.INFO, format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S' )
        self.logger = logging.getLogger('validation')
        self.logger.info("----------- Validation step started -----------")

    def _get_arguments(self):
        parser = argparse.ArgumentParser(description='Validation')
        parser.add_argument('-execDir','--execution_path', help='Directory where the logs and history will be saved', required=True)
        parser.add_argument('-paramFile','--parameter_file', help='Running configuration file', required=False)
        
        args = parser.parse_args()
        execdir = args.execution_path
        
        with open( args.parameter_file, 'r' ) as g:
            self.config = json.load(g)

        try:
            self.config_path = None
            self.flag_parallel = False
            if( 'config_hpc' in self.config ):
                self.config_path = self.config['config_hpc']
                self.flag_parallel = True

            self.cutoff_consensus = 0.8
            if( 'cutoff_consensus' in self.config ):
                self.cutoff_consensus = self.config['cutoff_consensus']
            
            self.outDir = self.config["outpath"]
            self.outPredDir = self.config["path_prediction_result"]

            self.__setup_logdir( execdir )
        except:
            raise Exception("Mandatory fields not found in config. file")
    
    def _setup_out_folders(self):
        self.out = os.path.join( self.outDir, 'validation' )
        if( not os.path.isdir( self.out ) ) :
            os.makedirs( self.out )
        
        self.processedCTDir = os.path.join( self.outDir, 'process_ct', 'processed_cts' )
        self.predInDir = os.path.join( self.outDir, 'input_prediction' )
        
        self.augdsDir = os.path.join( self.outDir, 'augmented_data' )
        if( not os.path.isdir( self.augdsDir ) ) :
            os.makedirs( self.augdsDir )
        
    def __load_mapping_pmid_nctid(self):
        mapp = {}
        opath = os.path.join( self.out, 'mapping_ct_pubmed.json' )
        if( not os.path.isfile(opath) ):
            ctids = set()
            for f in tqdm( os.listdir( self.inPredDir ) ):
                if( f.endswith('.txt') ):
                    pmid = f.split('_')[0]
                    if( pmid not in mapp ):
                        mapp[pmid] = set()
                    path = os.path.join( self.inPredDir, f)
                    abs = open(path).read()
                    tokens = abs.split(' ')
                    ncts = list( filter( lambda x: (x.find('NCT0') != -1), tokens ))
                    if( len(ncts) > 0 ):
                        for nid in ncts:
                            tr = re.findall( r'(NCT[0-9]+)', nid )
                            if( len(tr) > 0 ):
                                for t in tr:
                                    if( t.startswith('NCT') ):
                                        ctid = t
                                        mapp[pmid].add(ctid)
                                        ctids.add(ctid)
            for k in mapp:
                mapp[k] = list(mapp[k])
            json.dump( mapp, open(opath, 'w') )
            print('All ctids linked', len(ctids) ) # 51827
            print('All pubmeds linked', len(mapp) ) # 68830
        else:
            mapp = json.load( open(opath, 'r') )
            for k in mapp:
                mapp[k] = set(mapp[k])
        return mapp

    def __get_snippets_pred_labels(self, pmid):
        f = self.predictions_df[ self.predictions_df['pmid'] == pmid ]
        anns = []
        for i in f.index:
            label = str( self.predictions_df.loc[i, 'entity_group'])
            term = str( self.predictions_df.loc[i, 'word'])
            anns.append( [term, label] )
        
        return anns
    
    def _map_nctid_pmid_general(self, label_exp):
        mapp = self.__load_mapping_pmid_nctid()
        
        f = os.path.join( self.outPredDir, 'consensus_augmentation_models.tsv' )
        omap = os.path.join( self.out, f'general_mapping_{label_exp}_nct_pubmed.tsv')
        g = open( omap, 'w' )
        g.write( 'pmid\tctid\ttext\tlabel\n' )
        g.close()
        
        path = os.path.join( self.outPredDir, f)
        df = pd.read_csv(path, sep='\t')
        self.predictions_df = df
        del df
        
        lines = []
        for pmid in tqdm(mapp):
            anns = self.__get_snippets_pred_labels( pmid )
            cts = mapp[pmid]
            for ctid in cts:
                if( len(anns) > 0 ):
                    for a in anns:
                        items = [pmid, ctid]+a
                        if( len(items) == 4 ):
                            line = '\t'.join( items )
                            lines.append(line)
                            if(  len(lines) %1000 == 0 ):
                                with open( path_partial, 'a' ) as g:
                                    g.write( ('\n'.join(lines) )+'\n' )
                                lines = []

        if(  len(lines) > 0 ):
            with open( omap, 'a' ) as g:
                g.write( ('\n'.join(lines) )+'\n' )
    
    def __treat_predictions(self, path, fname):
        npath = os.path.join( os.path.dirname( path ), 'treated_'+fname )
        fn = open(npath, 'w')
        g = open(path, 'r')
        for line in g:
            l = line.replace("'",'').replace('"','')
            fn.write(l)
        g.close()
        fn.close()

        return npath

    def _map_nctid_pmid_general_parallel(self, label_exp):
        mapp = self.__load_mapping_pmid_nctid()
        for k in mapp:
            mapp[k] = list(mapp[k])

        f = os.path.join( self.outPredDir, 'consensus_augmentation_models.tsv' )
        fname = 'consensus'
        path = os.path.join( self.out, f'general_mapping_{label_exp}_nct_pubmed.tsv')
        if( not os.path.isfile(path) ):
            g = open( path, 'w' )
            g.write( 'pmid\tctid\ttext\tlabel\n' )
            g.close()

            inpath = os.path.join( self.outPredDir, f)
            treated = self.__treat_predictions(inpath, f)

            elements = []
            for pmid in tqdm(mapp):
                elements.append( [pmid, mapp, treated] )

            job_name = f"mapping_parallel_{fname}"
            job_path = os.path.join( self.out, job_name )
            chunk_size = 1000
            script_path = os.path.join(os.path.dirname( os.path.abspath(__file__)), '_aux_mapping.py')
            command = "python3 "+script_path
            config = self.config_path
            prepare_job_array( job_name, job_path, command, filetasksFolder=None, taskList=elements, chunk_size=chunk_size, ignore_check = True, wait=True, destroy=True, execpy='python3', hpc_env = 'slurm', config_path=config )

            test_path_partial = os.path.join( job_path, f'part-task-1.tsv' )
            if( os.path.exists(test_path_partial) ):
                path_partial = os.path.join( job_path, f'part-task-*.tsv' )
                cmdStr = 'for i in '+path_partial+'; do cat $i; done | sort -u >> '+path
                execAndCheck(cmdStr)

                cmdStr = 'for i in '+path_partial+'; do rm $i; done '
                execAndCheck(cmdStr)

    def _manage_mapping(self, label):
        self.logger.info("[Validation step] Task (Mapping predictions to pmid and CT ) started -----------")
        
        if( not self.flag_parallel ):
            self._map_nctid_pmid_general(label)
        else:
            self._map_nctid_pmid_general_parallel(label)

        self.logger.info("[Validation step] Task (Mapping predictions to pmid and CT ) ended -----------")
        
    def __load_cts_library(self, allids):
        pathout = os.path.join(self.out, 'ctlib.json')
        dat = {}
        if( os.path.isfile(pathout) ):
            dat = json.load( open(pathout,'r') )
        else:
            for _id in allids:
                path = os.path.join( self.processedCTDir, f"proc_ct_{_id}.json" )
                if( os.path.isfile(path) ):
                    dat[_id] = json.load( open(path, 'r') )

            json.dump( dat, open(pathout,'w') )

        return dat, pathout

    def __normalize_string(self, s):
        s = s.lower()
        s = ' '.join( re.findall(r'[a-zA-Z0-9\-]+',s) )
        return s

    def __send_query_fast(self, snippet, ctlib, ctid, label='all'):
        cutoff = 0.3
        results = []
        ct = ctlib[ctid]

        keys = list(ct)
        if(label != 'all'):
            tags = []
            if( label in keys ):
                tags = [label]
        else:
            tags = keys

        for k in tags:
            try:
                elements = [ ct[k] ]
                if( isinstance(ct[k], set) or isinstance(ct[k], list) ):
                    elements = ct[k]
                    
                for el in elements:
                    el = str(el)
                    clss = 'exact'
                    
                    nel = self.__normalize_string(el)
                    nsnippet = self.__normalize_string(snippet)
                    
                    score = Levenshtein.ratio( nsnippet, nel )
                    if(score >= cutoff):
                        if( score < 1):
                            clss = 'm'+str(score).split('.')[1][0]+'0'
                        results.append( { 'hit': el, 'ct_label': k, 'score': f'{score}-{clss}' } )
            except:
                pass

        return results

    def _get_predictions(self, sourcect, ctlib, pathlib, label_result='', mode='fast' ):
        cts_available = set(ctlib)

        res = os.path.join( self.out, f'{label_result}_results_test_validation.tsv')
        gone = set()
        if( os.path.isfile(res) ):
            df = pd.read_csv( res, sep='\t' )
            for i in tqdm(df.index):
                ctid = df.loc[i, 'ctid']
                pmid = df.loc[i, 'pmid']
                test_text = df.loc[i, 'test_text']
                test_label = df.loc[i, 'test_label']

                line = f"{ctid}\t{pmid}\t{test_label}\t{test_text}"
                gone.add(line)
        else:
            f = open(res, 'w')
            f.write("ctid\tpmid\ttest_label\tfound_ct_label\ttest_text\tfound_ct_text\tscore\n")
            f.close()

        print('Skipped', len(gone))
        print('cts available', len(cts_available) )
        lines = []
        idx = 0
        k = 10000
        df = pd.read_csv( sourcect, sep='\t' )

        for i in tqdm(df.index):
            ctid = df.loc[i, 'ctid']
            pmid = df.loc[i, 'pmid']
            test_text = str( df.loc[i, 'text'] )
            test_label = df.loc[i, 'label']

            aux = f"{ctid}\t{pmid}\t{test_label}\t{test_text}"
            if( (ctid in cts_available) and (not aux in gone) ):
                gone.add(aux)
                results = self.__send_query_fast( test_text, ctlib, ctid, label=test_label )
                
                for r in results:
                    found_ct_text = r['hit']
                    found_ct_label = r['ct_label']
                    score = r['score']
                    line = f"{ctid}\t{pmid}\t{test_label}\t{found_ct_label}\t{test_text}\t{found_ct_text}\t{score}"

                    lines.append(line)
                    if(idx%k==0 and len(lines)>0 ):
                        with open(res, 'a') as g:
                            g.write( ('\n'.join(lines))+'\n' )
                        lines = []

        if( len(lines)>0 ):
            with open(res, 'a') as g:
                g.write( ('\n'.join(lines))+'\n' )

    def _get_predictions_parallel(self, sourcect, ctlib, pathlib, label_result='', mode='fast'):
        result_path = os.path.join( self.out, f'{label_result}_results_test_validation.tsv')
        gone = set()
        if( os.path.isfile(result_path) ):
            df = pd.read_csv( result_path, sep='\t' )
            for i in tqdm(df.index):
                ctid = df.loc[i, 'ctid']
                pmid = df.loc[i, 'pmid']
                test_text = df.loc[i, 'test_text']
                test_label = df.loc[i, 'test_label']

                line = f"{ctid}\t{pmid}\t{test_label}\t{test_text}"
                gone.add(line)
        else:
            f = open(result_path, 'w')
            f.write("ctid\tpmid\ttest_label\tfound_ct_label\ttest_text\tfound_ct_text\tscore\n")
            f.close()

        elements = []

        df = pd.read_csv( sourcect, sep='\t' )
        for i in tqdm(df.index):
            ctid = df.loc[i, 'ctid']
            pmid = df.loc[i, 'pmid']
            test_text = df.loc[i, 'text']
            test_label = df.loc[i, 'label']

            aux = f"{ctid}\t{pmid}\t{test_label}\t{test_text}"
            elements.append( [ctid, pmid, test_text, test_label] )

        flag_ollama = False

        job_name = f"{mode}_prediction_parallel"
        job_path = os.path.join( self.out, job_name )
        chunk_size = 10000
        script_path = os.path.join(os.path.dirname( os.path.abspath(__file__)), '_aux_prediction.py')
        command = f"python3 {script_path} {pathlib} {mode}"
        config = self.config_path
        prepare_job_array( job_name, job_path, command, filetasksFolder=None, taskList=elements, chunk_size=chunk_size, ignore_check = True, wait=True, destroy=True, execpy='python3', hpc_env = 'slurm', config_path=config, flag_ollama = flag_ollama, ncpus=8 )

        test_path_partial = os.path.join( job_path, f'part-task-1.tsv' )
        if( os.path.exists(test_path_partial) ):
            path_partial = os.path.join( job_path, f'part-task-*.tsv' )
            cmdStr = 'for i in '+path_partial+'; do cat $i; done | sort -u >> '+result_path
            execAndCheck(cmdStr)

            cmdStr = 'for i in '+path_partial+'; do rm $i; done '
            execAndCheck(cmdStr)

    def __aggregate_nctids(self):
        allids = set()
        f = os.path.join( self.out, f'general_mapping_{label_exp}_nct_pubmed.tsv')
        sourcect = os.path.join( self.out, f)
        df = pd.read_csv( sourcect, sep='\t' )
        ctids = set(df.ctid.unique())
        allids = allids.union(ctids)
        return allids

    def _manage_prediction(self, label_exp):
        self.logger.info("[Validation step] Task (Obtaining predictions ) started -----------")
        
        widectids = self.__aggregate_nctids()
        ctlib, pathlib = self.__load_cts_library(widectids)

        f = os.path.join( self.out, f'general_mapping_{label_exp}_nct_pubmed.tsv')
        sourcect = os.path.join( self.out, f)

        if(flag_parallel):
            self._get_predictions_parallel(sourcect, ctlib, pathlib, f'{label_exp}', 'fast' )
        else:
            self._get_predictions(sourcect, ctlib, pathlib, f'{label_exp}', 'fast' )

        self.logger.info("[Validation step] Task (Obtaining predictions ) ended -----------")
        
    def __load_mapped_positions(self):
        path = os.path.join( self.outPredDir, "consensus_augmentation_models.tsv" )
        df = pd.read_csv( path, sep='\t' )

        mapped_positions = {}
        opath = os.path.join( self.out, "consensus_mapped.json")
        if( os.path.exists(opath) ):
            mapped_positions = json.load( open(opath, 'r') )
        else:
            if( isinstance(df, str) ):
                df = pd.read_csv(df, sep='\t')

            for i in df.index:
                ofile = df.loc[i, 'input_file'] 
                pmid = df.loc[i, 'input_file'].split('_')[0]
                entity = df.loc[i, 'entity_group']
                word = df.loc[i, 'word']
                try:
                    start = int( df.loc[i, 'start'] )
                    end = int( df.loc[i, 'end'] )

                    key = f'{pmid}#$@{entity}#$@{word}'
                    if( not key in mapped_positions ):
                        mapped_positions[key] = []
                    mapped_positions[key].append( { 'start': start, 'end': end, 'input_file': ofile } )
                except:
                    pass
            json.dump(mapped_positions, open(opath, 'w') )

        return mapped_positions

    def _aggregate_report_add_info(self, label_result):
        self.logger.info("[Validation step] Task (Grouping predictions, ranking, and aggregating positional information ) started -----------")
        
        cutoff_consensus = self.cutoff_consensus

        mapped_positions = self.__load_mapped_positions()

        path = os.path.join( self.out, f'{label_result}_results_test_validation.tsv')
        result_path = os.path.join( self.out, f'grouped_{label_result}_results_validation.tsv')
        
        if( not os.path.isfile(result_path) ):
            rdf = pd.read_csv( path, sep='\t')
            rdf = rdf[ ['ctid', 'pmid','test_label','test_text', 'score'] ]
            
            rdf['val'] = [ float(s.split('-')[0]) for s in rdf['score'] ]
            rdf['stat_class'] = [ s.split('-')[1] for s in rdf['score'] ]
            rdf = rdf.drop('score', axis=1)
            rdf = rdf.groupby(['ctid', 'pmid','test_label','test_text']).max().reset_index()

            input_ = []
            start_ = []
            end_ = []
            for i in rdf.index:
                pmid = rdf.loc[i, 'pmid']
                entity = rdf.loc[i, 'test_label']
                word = rdf.loc[i, 'test_text']

                key = f'{pmid}#$@{entity}#$@{word}'
                pos = mapped_positions[key][0]
                input_.append(pos["input_file"])
                start_.append(pos["start"])
                end_.append(pos["end"])

            rdf["input_file"] = input_
            rdf["start"] = start_
            rdf["end"] = end_

            rdf = rdf[ ["input_file", "start", "end", "pmid", "test_label", "test_text", "val", "stat_class"] ]
            rdf.columns = ["input_file", "start", "end", "pmid", "entity", "word", "val", "stat_class"]
            result_path = os.path.join( self.out, f'grouped_{label_result}_results_validation.tsv')
            rdf.to_csv( result_path, sep='\t', index=None )
            
            rdf = rdf[ rdf['val'] >= cutoff_consensus ]
            rdf = rdf.sort_values( by = "val", ascending = False )
            result_path = os.path.join( self.out, f'top_grouped_{label_result}_results_validation.tsv')
            rdf.to_csv( result_path, sep='\t', index=None )
        else:
            rdf = pd.read_csv( result_path, sep='\t')

        self.logger.info("[Validation step] Task (Grouping predictions, ranking and aggregating positional information ) ended -----------")
        
    def _create_annotation_txt_per_section(self, label_aux, cutoff_consensus, force_rewrite=False ):
        self.logger.info("[Validation step] Task (Creating final annotated corpus and text files ) started -----------")
                
        folder_out = self.augdsDir
        oconsensus = os.path.join( self.out, f'top_grouped_{label_result}_results_validation.tsv' )
        
        cnt = {}
        not_found = set()
        df = pd.read_csv(oconsensus, sep='\t')
        for i in df.index:
            pmid = df.loc[i, 'pmid']
            start = df.loc[i, 'start']
            end = df.loc[i, 'end']
            entity = df.loc[i, 'entity']
            word = df.loc[i, 'word']
            outname = df.loc[i, 'input_file']

            # Feed annotation files that will be the input for another training round
            path = os.path.join( folder_out, f'{outname}.ann')
            if(not outname in cnt):
                cnt[outname] = 1
                g = open(path, 'w')
                g.close()

            with open(path, 'a') as g:
                g.write( f"T{ cnt[outname] }\t{entity} {start} {end}\t{word}\n" )
            cnt[outname] += 1

            # Check txt original file
            inpath = os.path.join(self.inPredDir, f"{outname}.txt")
            outpath = os.path.join( folder_out, f'{outname}.txt')
            if( not os.path.isfile(outpath) or force_rewrite ):
                if( not os.path.isfile(inpath) ):
                    inpath = inpath.replace('.txt', '..txt')

                text = open(inpath, 'r').read()
                with open(outpath, 'w') as g:
                    g.write( text.strip() )

        opath = os.path.join( self.out, "keys_not_mapped_result4.txt")
        f = open(opath, 'w')
        f.write( '\n'.join(not_found) )
        f.close()

        self.logger.info("[Validation step] Task (Creating final annotated corpus and text files ) ended -----------")
                     
    def _mark_as_done(self):
        f = open( self.fready, 'w')
        f.close()

        self.logger.info("----------- Validation ended -----------")
    
    def run(self):
        label = 'predlev'
        self._manage_mapping(label)

        self._manage_prediction(label)

        self._aggregate_report_add_info(label)

        self._create_annotation_txt_per_section(label)

        self._mark_as_done()
