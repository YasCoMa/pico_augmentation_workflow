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
    def __init__(self, outdir):
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
        
    def __check_stopwords( self, term):
        flag = True
        for st in self.stopwords:
            if( term.lower().find(st) != -1 ):
                return False
        
        return flag

    def __treat_eligibility(self, s):
        inc = []
        exc = []
        sex = set()
        age = set()
        try:
            e = s['protocolSection']['eligibilityModule']['eligibilityCriteria']
            
            parts = e.split('Exclusion Criteria:')
            inc = []
            conds = list( filter( lambda x: x!= "", parts[0].split('\n') ))
            conds = list( map( lambda x: x.replace('\\>', '>').replace('\\<', '<').replace('\\^', '^').replace('≤', '<=').replace('≥', '>=').replace('\\[', '[').replace('\\]', ']'), conds ))
            conds = list( map( lambda x: re.sub('(\*+) |([0-9]+)\. ','', x ).strip(), conds ))
            conds = list( filter( lambda x: ( not x.endswith(':') and self.__check_stopwords(x) ), conds ))
            inc = conds
            
            exc = []
            if( len(parts) > 1 ):
                conds = list( filter( lambda x: x!= "", parts[1].split('\n') ))
                conds = list( map( lambda x: x.replace('\\>', '>').replace('\\<', '<').replace('\\^', '^').replace('≤', '<=').replace('≥', '>=').replace('\\[', '[').replace('\\]', ']'), conds ))
                conds = list( map( lambda x: re.sub('(\*+) |([0-9]+)\. ','', x ).strip(), conds ))
                conds = list( filter( lambda x: ( not x.endswith(':') and self.__check_stopwords(x) ), conds ))
                exc = conds
            
        except:
            pass
            
        try:
            keys = ['minimumAge', 'maximumAge']
            for k in keys:
                if(k in s['protocolSection']['eligibilityModule']):
                    age.add( s['protocolSection']['eligibilityModule'][k])
            if( 'sex' in s['protocolSection']['eligibilityModule']):
                gender = s['protocolSection']['eligibilityModule']['sex']
                if( gender.lower() == 'all' ):
                    sex.add( 'male' )
                    sex.add( 'female' )
                    sex.add('ALL')
                else:
                    sex.add( gender )
            
        except:
            pass
        
        return [inc, exc, sex, age]
    
    def __get_ct_info(self, s ):
        _id = s["protocolSection"]["identificationModule"]["nctId"]
        path = os.path.join( self.processedCTDir, f"proc_ct_{_id}.json" )
        if( not os.path.isfile(path) ):
            clabels = set()
            ilabels = set()
            cids = set()
            itids = set()
            control_groups = set()
            interv_groups = set()
            
            condition = set()
            out_measures = set()
            eligibility = set()
            outcomes = set()
            location = set()
            age = set()
            gender = set()
            ethnicity = set()
            
            totp = -1
            controlp = -1
            intervp = -1
            
            # For P entities
            try:
                mintervs = s['protocolSection']['armsInterventionsModule']
                groups = mintervs['armGroups']
                clabels = set()
                ilabels = set()
                for g in groups:
                    if( g['type'] == 'NO_INTERVENTION' ):
                        clabels.add( g['label'] )
                        control_groups.add( g['label'] )
                    else:
                        ilabels.add( g['label'] )
            except:
                pass
            
            params = self._treat_eligibility(s)
            eligibility.update( params[0]+params[1] )
            inclusion = params[0]
            exclusion = params[1]
            gender.update( list(params[2]) )
            age.update( list(params[3]) )
            
            try:
                mconds = s['protocolSection']['conditionsModule']
                condition.update( mconds['conditions'] )
            except:
                pass
            
            try:
                locs = s['protocolSection']['contactsLocationsModule']['locations']
                for l in locs:
                    location.update( [ l[k] for k in ['facility', 'city', 'country'] ] )
            except:
                pass
            
            try:
                moutcomes = s['protocolSection']['outcomesModule']
                for k in moutcomes:
                    outs = moutcomes[k]
                    for g in outs:
                        outcomes.add( g['measure'] )
            except:
                pass
            
            try:
                moutcomes = s['resultsSection']['outcomeMeasuresModule']['outcomeMeasuresModule']['outcomeMeasures']
                for g in moutcomes:
                    out_measures.add( g['title'] )
            except:
                pass
            
            try:
                mintervs = s['protocolSection']['armsInterventionsModule']
                interventions = mintervs['interventions']
                for g in interventions:
                      interv_groups.add( g['name'] )
                      interv_groups.update( g['otherNames'] )
            except:
                pass
                
            try:
                gresults = s['resultsSection']['baselineCharacteristicsModule']['measures']
                for g in gresults:
                    if( g['title'].find('ethnicity') != -1 ):
                        ethnicity.update( list(map( lambda x: x['title'], g['classes'] )) )
            except:
                pass
            
            # To classify the results numbers in iv or cv
            totid = -1  
            try:
                gresults = s['resultsSection']['baselineCharacteristicsModule']['groups']
                totid = gresults[0]['id']
                for g in gresults:
                    name = g['title']
                    _id = g['id']
                    if( name in clabels ):
                        control_groups.add(name)
                        cids.add(_id)
                    else:
                        if( name.lower() != 'total'):
                            interv_groups.add(name)
                            itids.add(_id)
                        else:
                            totid = _id
            except:
                pass   
            
            try:
                gresults = s['resultsSection']['baselineCharacteristicsModule']['denoms']
                for g in gresults:
                    gid = g['groupId']
                    if( gid in clabels ):
                        controlp += int(g['value'])
                        
                    if( gid == totid ):
                        totp = int(g['value'])
                intervp = totp - controlp
            except:
                pass     
            
            # abs or percent values => bin
            # mean, median, sd, q1, q3 => cont  
            md = {}
            try:
                arr = s['resultsSection']['outcomeMeasuresModule']['outcomeMeasures']
                for it in arr:
                    ntype = 'bin'
                    spec = 'abs'
                    if( it['unitOfMeasure'].lower().find('number')==-1 and it['unitOfMeasure'].lower().find('percentage')==-1 and it['unitOfMeasure'].lower().find('count')==-1 ):
                        ntype = 'cont'
                        spec = it['paramType'].lower()
                    else:
                        if( it['unitOfMeasure'].lower().find('percentage')==-1 ):
                            spec = 'percent'
                        
                    ms = it['classes'][0]['categories'][0]['measurements']
                    for m in ms:
                        val = m['value']
                        gr = 'iv'
                        if( m['groupId'] in cids ):
                            gr='cv'
                        key = f'{gr}-{ntype}-{spec}'
                        if( not key in md):
                            md[key]=set()
                        md[key].add(val) 
            except:
                pass

            # Participants
            if( totp != -1 ):
                md['total-participants'] = totp
            if( intervp != -1 ):
                md['intervention-participants'] = intervp
            if( controlp != -1 ):
                md['control-participants'] = controlp
            if( len(age) > 0 ):
                md['age'] = age
            if( len(eligibility) > 0 ):
                md['eligibility'] = eligibility
            if( len(ethnicity) > 0 ):
                md['ethnicity'] = ethnicity
            if( len(condition) > 0 ):
                md['condition'] = condition
            if( len(location) > 0 ):
                md['location'] = location
            
            # Intervention & Control
            if( len(control_groups) > 0 ):
                md['control'] = control_groups
            if( len(interv_groups) > 0 ):
                md['intervention'] = interv_groups
            
            # Outcome
            if( len(outcomes) > 0 ):
                md['outcome'] = outcomes
            if( len(out_measures) > 0 ):
                md['outcome-Measure'] = out_measures
        
            aux = {}
            for k in md:
                if( isinstance( md[k], set ) ):
                    aux[k] = list(md[k])
                else:
                    aux[k] = md[k]
            json.dump( aux, open(path, 'w') )
        else:
            md = json.load( open(path, 'r') )
        
        return md

    def _parse_ct_raw(self, ids):
        self.logger.info("[Process CT step] Task (Parsing CT to extract target entity true values ) started -----------")
        
        for f in tqdm( os.listdir(self.ctDir) ):
            if( f.startswith('raw') ):
                path = os.path.join( self.ctDir, f )
                dt = json.load( open( path, 'r' ) )
                for s in dt:
                    _ = self.__get_ct_info()

        self.logger.info("[Process CT step] Task (Parsing CT to extract target entity true values ) ended -----------")
        
    def _mark_as_done(self):
        f = open( self.fready, 'w')
        f.close()

        self.logger.info("----------- Process CT ended -----------")
    
    def run(self):
        self._get_clinical_trials()
        self._make_mapping_ct_pubmed()
        self._parse_ct_raw()
        self._mark_as_done()
        
if( __name__ == "__main__" ):
    odir = './out'
    i = InformationExtractionCT( odir )
    i.run()
