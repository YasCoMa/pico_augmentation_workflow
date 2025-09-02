import os
import re
import json
import time
from tqdm import tqdm
import requests
import pandas as pd

import logging
import argparse

class InformationExtractionCT:
    def __init__(self):
        self.stopwords = ["other", "key", "inclusion criteria", "exclusion criteria", "not specified", "see disease characteristics"]
        
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(filename='log_preprocess_ct.log', level=logging.INFO)
        
        self._get_arguments()
        self._setup_out_folders()

        self.fready = os.path.join( self.logdir, "tasks_process_ct.ready")
        self.ready = os.path.exists( self.fready )
        if(self.ready):
            self.logger.info("----------- Process CT step skipped since it was already computed -----------")
            self.logger.info("----------- Process CT step ended -----------")
        
    def __setup_logdir(self, execdir):
        self.logdir = os.path.join( execdir, "logs" )
        if( not os.path.exists(self.logdir) ):
            os.makedirs( self.logdir )

        logf = os.path.join( self.logdir, "process_ct.log" )
        logging.basicConfig( filename=logf, encoding="utf-8", filemode="a", level=logging.INFO, format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S' )
        self.logger = logging.getLogger('process_pubmed')
        self.logger.info("----------- Prediction step started -----------")

    def _get_arguments(self):
        parser = argparse.ArgumentParser(description='Process CT')
        parser.add_argument('-execDir','--execution_path', help='Directory where the logs and history will be saved', required=True)
        parser.add_argument('-paramFile','--parameter_file', help='Running configuration file', required=False)
        
        args = parser.parse_args()
        execdir = args.execution_path
        
        with open( args.parameter_file, 'r' ) as g:
            self.config = json.load(g)

        try:
            self.outDir = self.config["outpath"]

            self.__setup_logdir( execdir )
        except:
            raise Exception("Mandatory fields not found in config. file")
    
    def _setup_out_folders(self):
        self.out = os.path.join( self.outDir, 'process_ct' )
        if( not os.path.isdir( self.out ) ) :
            os.makedirs( self.out )
        
        self.ctDir = os.path.join( self.out, 'clinical_trials' )
        if( not os.path.isdir( self.ctDir ) ) :
            os.makedirs( self.ctDir )

        self.processedCTDir = os.path.join( self.out, 'processed_cts' )
        if( not os.path.isdir( self.processedCTDir ) ) :
            os.makedirs( self.processedCTDir )
            
    def _get_clinical_trials(self):
        self.logger.info("[Process CT step] Task (Getting raw complete clinical trial from api) started -----------")
        
        outp = self.ctDir
        
        root = "https://clinicaltrials.gov/api/v2/studies?filter.overallStatus=COMPLETED&countTotal=true&pageSize=1000"
        r = requests.get(root)
        dat = r.json()
        tgone = 0
        p = 0
        lobj = []
        total = dat["totalCount"]

        ns = dat["nextPageToken"]
        for s in tqdm(dat['studies']):
            lobj.append(s)
                
        tgone += 1000
        p += 1
        ofile = os.path.join( outp, f"raw_completed_cts_page-{ p }.json" )
        json.dump( lobj, open(ofile, 'w') )
            
        while( ns != None and ns != '' ):
            lobj = []
            r = requests.get( root+'&pageToken='+ns )
            dat = r.json()
            ns = None
            if( "nextPageToken" in dat ):
                ns = dat["nextPageToken"]
                
            for s in tqdm(dat['studies']):
                lobj.append(s)
                
            tgone += 1000
            p += 1
            ofile = os.path.join( outp, f"raw_completed_cts_page-{ p }.json" )
            json.dump( lobj, open(ofile, 'w') )
            print(ns, tgone, total)
            time.sleep(1)

        self.logger.info("[Process CT step] Task (Getting raw complete clinical trial from api) ended -----------")
        
    def _make_mapping_ct_pubmed(self):
        self.logger.info("[Process CT step] Task (Mapping CT IDs to their respective references (PubMed ID) ) started -----------")
        
        all_refs = set()
        omap = os.path.join( self.out, 'complete_mapping_pubmed.tsv' )
        gone = set()
        if( os.path.isfile(omap) ):
            fm = open( omap, 'r' )
            for line in fm:
                l = line.replace('\n','')
                if( (len(l) > 2) and (not l.startswith('pmid') ) ):
                    gone.add( l.split('\t')[0] )
                    all_refs.add( l.split('\t')[1] )
            fm.close()
        else:
            fm = open( omap, 'w' )
            fm.write("ctid\tpmid\n")
            fm.close()
        
        files = list( filter( lambda x: x.startswith('raw'), os.listdir( self.ctDir ) ))
        for f in tqdm( files ) :
            path = os.path.join( self.ctDir, f )
            dt = json.load( open( path, 'r' ) )
            for s in dt:
                _id = s['protocolSection']['identificationModule']['nctId']
                if( not _id in gone ):
                    try:
                        pmids = set()
                        ref = s['protocolSection']['referencesModule']['references']
                        for r in ref:
                            try:
                                pmids.add(r['pmid'])
                                all_refs.add(r['pmid'])
                            except:
                                pass
                        for r in pmids:
                            with open(omap, 'a') as fm:
                                fm.write(f"{_id}\t{r}\n")
                    except:
                        pass

        self.logger.info("[Process CT step] Task (Mapping CT IDs to their respective references (PubMed ID) ) ended -----------")
        
     
    def _mark_as_done(self):
        f = open( self.fready, 'w')
        f.close()

        self.logger.info("----------- Process CT ended -----------")
    
    def run(self):
        #self._get_clinical_trials()
        #self._make_mapping_ct_pubmed()
        #self._parse_ct_raw()
        self._mark_as_done()
        
if( __name__ == "__main__" ):
    odir = './out'
    i = InformationExtractionCT( )
    i.run()
