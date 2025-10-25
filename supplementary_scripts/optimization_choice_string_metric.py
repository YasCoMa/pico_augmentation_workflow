import os
import re
import sys
import json
import pickle
import optuna
import shutil
import pandas as pd
import Levenshtein
from tqdm import tqdm
from strsimpy import *
import plotly.express as px
from scipy.stats import ranksums

root_path = (os.path.sep).join( os.path.dirname(os.path.realpath(__file__)).split( os.path.sep )[:-1] )
sys.path.append( root_path )
from supplementary_scripts.similarity_metrics import *

gold_pairs_test = ""

def objective_distance(trial):
    metrics = ['levenshtein', 'damerau', 'jaccard', 'cosine', 'jaro_winkler', 'longest_common_subsequence', 'metric_lcs', 'ngram', 'optimal_string_alignment', 'overlap_coefficient', 'qgram', 'sorensen_dice']
    norm = trial.suggest_categorical("normalization", ['no', 'yes'])
    m = trial.suggest_categorical("metric", metrics)
    
    scores = []
    df = pd.read_csv( gold_pairs_test, sep='\t')
    tmp = df[ ['ctid', 'pmid', 'test_text', 'test_label'] ]
    df = df[ ['found_ct_text', 'test_text'] ]
    for i in df.index:
        a, b = process_pair( df.loc[i, 'found_ct_text'], df.loc[i, 'test_text'], norm )
        try:
            dist = eval(f"compute_distance_{m}")(a, b)
        except:
            dist = 'invalid'
            pass
        scores.append(dist)
    tmp['score'] = scores
    tmp = tmp[ tmp.score != 'invalid' ]

    tmp = tmp.groupby( ['ctid', 'pmid', 'test_text', 'test_label'] ).min().reset_index()
    mean = tmp.score.mean()
    
    return mean

def objective_similarity(trial):
    metrics = ['levenshtein', 'damerau', 'jaccard', 'cosine', 'jaro_winkler', 'longest_common_subsequence', 'metric_lcs', 'ngram', 'optimal_string_alignment', 'overlap_coefficient', 'qgram', 'sorensen_dice']
    norm = trial.suggest_categorical("normalization", ['no', 'yes'])
    m = trial.suggest_categorical("metric", metrics)

    scores = []
    df = pd.read_csv( gold_pairs_test, sep='\t')
    tmp = df[ ['ctid', 'pmid', 'test_text', 'test_label'] ]
    df = df[ ['found_ct_text', 'test_text'] ]
    for i in df.index:
        a, b = process_pair( df.loc[i, 'found_ct_text'], df.loc[i, 'test_text'], norm )
        try:
            dist = eval(f"compute_similarity_{m}")(a, b)
        except:
            dist = 'invalid'
            pass
        scores.append(dist)
    tmp['score'] = scores
    tmp = tmp[ tmp.score != 'invalid' ]

    mean = 0
    if( len(tmp) > 0 ): # Not all the metrics have the similarity function implemented
        tmp = tmp.groupby( ['ctid', 'pmid', 'test_text', 'test_label'] ).max().reset_index()
        mean = tmp.score.mean()
    
    return mean

class SimilarityMetricOptimization:
    def __init__(self, fout):
        self.out = fout
        
        # Uncompress the original human curated dataset to replicate analysis
        self.goldDir = os.path.join( fout, "pico_corpus_brat_annotated_files" ) 
        if( not os.path.exists(self.goldDir) ):
            processed_goldds = os.path.join(root_path, "supplementary_scripts", "pico_corpus.tar.gz")
            os.system(f"tar xzf {processed_goldds} -C {self.out}")

        # Uncompress the json files regarding the NCBI clinical trials raw data
        self.ctlib = os.path.join( fout, "ctlib.json" ) 
        if( not os.path.exists(self.ctlib) ):
            ctlibPath = os.path.join(root_path, "supplementary_scripts", "ctlib.json")
            shutil.copy(ctlibPath, self.ctlib)

        self.fout = os.path.join(fout, 'optimization')
        if( not os.path.isdir( self.fout ) ) :
            os.makedirs( self.fout )
            
    # -------- Gold ds

    def _get_snippets_labels(self, pmid):
        anns = []
        f = f"{pmid}.ann"
        path = os.path.join( self.goldDir, f)
        f = open(path, 'r')
        for line in f:
            l = line.replace('\n','').split('\t')
            label = l[1].split(' ')[0]
            term = l[2]
            anns.append( [term, label] )
        f.close()
        
        return anns
    
    def _map_nctid_pmid_gold(self):
        omap = os.path.join( self.out, 'goldds_labelled_mapping_nct_pubmed.tsv')
        f = open( omap, 'w' )
        f.write( 'pmid\tctid\ttext\tlabel\n' )
        f.close()
        
        for f in os.listdir( self.goldDir ):
            if( f.endswith('.txt') ):
                pmid = f.split('.')[0]
                
                path = os.path.join( self.goldDir, f)
                abs = open(path).read()
                tokens = abs.split(' ')
                ncts = list( filter( lambda x: (x.find('NCT0') != -1), tokens ))
                if( len(ncts) > 0 ):
                    for nid in ncts:
                        tr = re.findall( r'(NCT[0-9]+)', nid )
                        if( len(tr) > 0 ):
                            ctid = tr[0]
                            anns = self._get_snippets_labels( pmid )
                            for a in anns:
                                line = '\t'.join( [pmid, ctid]+a )
                                with open( omap, 'a' ) as g:
                                    g.write( line+'\n' )

    def __load_cts_library(self, allids):
        pathout = self.ctlib
        dat = {}
        if( os.path.isfile(pathout) ):
            dat = json.load( open(pathout,'r') )
        else:
            raise Exception("CTlib was not correctly copied")

        return dat, pathout

    def __normalize_string(self, s):
        s = str(s).lower()
        s = ' '.join( re.findall(r'[a-zA-Z0-9\-]+',s) )
        return s

    def _send_query_fast(self, snippet, ctlib, ctid, label='all'):
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
                    
                    #score = Levenshtein.ratio( nsnippet, nel )
                    score = cosine.Cosine(1).similarity( nsnippet, nel )
                    if(score >= cutoff):
                        if( score < 1):
                            clss = 'm'+str(score).split('.')[1][0]+'0'
                        results.append( { 'hit': el, 'ct_label': k, 'score': f'{score}-{clss}' } )
            except:
                pass

        return results

    def _get_predictions(self, sourcect, ctlib, pathlib, label_result='' ):
        cts_available = set(ctlib)

        res = os.path.join( self.out, f'{label_result}_results_test_validation.tsv')
        gone = set()
        '''
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
            '''
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
                results = self._send_query_fast( test_text, ctlib, ctid, label=test_label )
                
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

    def perform_validation_gold(self):
        # Map the Clinical rials ncbi data to the pubmed articles found in the human-curated abstracts
        # Obtain the pairwise combination between each human labeled annotation to the truth api data of the same label

        self._map_nctid_pmid_gold()
        sourcect = os.path.join( self.out, 'goldds_labelled_mapping_nct_pubmed.tsv')

        df = pd.read_csv( sourcect, sep='\t' )
        ids = set(df.ctid.unique())
        ctlib, pathlib = self.__load_cts_library(ids)

        self._get_predictions(sourcect, ctlib, pathlib, 'fast_gold' )
        global gold_pairs_test
        gold_pairs_test = os.path.join( self.out, "fast_gold_results_test_validation.tsv" )

    def check_best_string_sim_metric(self):
        # Perform the optuna optimization of string treatment and string similarity/distance metric
        # there are hard coded paths in the objective functions that must be changed

        studies = { 'similarity': 'maximize', 'distance': 'minimize' }
        for s in studies:
            opath = f"{self.fout}/by_{s}_best_params.pkl"
            if( not os.path.exists(opath) ):
                print("Optimizing for ", s)
                func = eval( f'objective_{s}' )
                direction = studies[s]

                study = optuna.create_study( direction = direction )
                study.optimize( func, n_trials = 50 )
                print('\tBest params:', study.best_trial)

                file = open( opath, "wb")
                pickle.dump(study.best_trial, file)
                file.close()

    def get_coverage_gold_ctapi(self):
        # Perform analysis of coverage and summarizes the optimization results in a boxplot
        opath = f"{self.fout}/by_similarity_best_params.pkl"
        bmetric = pickle.load( open( opath, 'rb') )
        bmetric = bmetric.params['metric']

        label_result = "fast_gold"

        path = os.path.join( self.out, f'{label_result}_results_test_validation.tsv')
        df = pd.read_csv( path, sep='\t')
        print("Number of CTs:", len(df.ctid.unique()) ) # 117
        print("Number of PMIDs:", len(df.pmid.unique()) ) # 129

        # transforming results
        scores_nyes = []
        scores_nno = []
        entities = []
        opath = os.path.join( self.fout, f'{label_result}_{bmetric}_results_validation.tsv')
        df = pd.read_csv( path, sep='\t')
        for i in df.index:
            entities.append( df.loc[i, 'test_label'] )

            ay, by = process_pair( df.loc[i, 'found_ct_text'], df.loc[i, 'test_text'], 'yes' )
            an, bn = process_pair( df.loc[i, 'found_ct_text'], df.loc[i, 'test_text'], 'no' )
            func = eval( f"compute_similarity_{bmetric}")
            try:
                sim_nyes = func(ay, by)
                sim_nno = func(an, bn)
            except:
                sim_nyes = 0
                sim_nno = 0
            scores_nyes.append(sim_nyes)
            scores_nno.append(sim_nno)

        df['score_with_norm'] = scores_nyes
        df['score_without_norm'] = scores_nno
        df.to_csv( opath, sep='\t', index=None)
        
        # grouping the results leaving only the best similarity value for each annotation candidate
        opath = os.path.join( self.fout, f'grouped_{label_result}_{bmetric}_results_validation.tsv')
        df = df[ ['ctid', 'pmid', 'test_text', 'test_label', 'score_with_norm', 'score_without_norm'] ]
        df = df.groupby( ['ctid', 'pmid', 'test_text', 'test_label'] ).max().reset_index()
        df.to_csv( opath, sep='\t', index=None)
        
        # plotting the distribution of similarity values (boxplot + data points) with normalization using the best metric found by the optimization
        df = df[ ['test_label', 'score_with_norm'] ]
        df.columns = ["Entity", f"{ bmetric.capitalize() } similarity"]
        fig = px.box(df, x="Entity", y=f"{ bmetric.capitalize() } similarity", points="all")
        opath = os.path.join(self.fout, f'{bmetric}_gold_grouped_distribution_scoresim.png')
        fig.write_image( opath )

        # plotting the distribution of similarity values (only boxplot) with their comparison with and without string normalization
        subdf = pd.DataFrame()
        subdf["Entity"] = entities * 2
        subdf[ f"{ bmetric.capitalize() } similarity" ] = scores_nyes + scores_nno

        pvalue = ranksums(scores_nyes, scores_nno).pvalue
        title = "Transformation (P-value: %.5f)" %(pvalue)
        subdf[ title ] = ['With normalization']*len(scores_nyes) + ['Without normalization']*len(scores_nno)
        fig = px.box( subdf, x="Entity", y=f"{ bmetric.capitalize() } similarity", color = title )
        opath = os.path.join(self.fout, f'{bmetric}_gold_all_distribution_scoresim.png')
        fig.write_image( opath )

    def run(self):
        self.perform_validation_gold()
        self.check_best_string_sim_metric()
        self.get_coverage_gold_ctapi()

if( __name__ == "__main__" ):
    odir = './out_ss_choice_optimization'
    if( len(sys.argv) > 1 ):
        odir = sys.argv[1]

    i = SimilarityMetricOptimization( odir )
    i.run()