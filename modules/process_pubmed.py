import os
import re
import json
import pandas as pd
from tqdm import tqdm
from Bio import Entrez
import xml.etree.ElementTree as ET

import logging
import argparse

class ProcessPubmed:
    def __init__(self):
        Entrez.email = 'ycfrenchgirl2@gmail.com'
        Entrez.api_key="4543094c8a41e6aecf9a1431bff42cfac209"
        
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(filename='log_preprocess_pubmed.log', level=logging.INFO)
        
        self._get_arguments()
        self._setup_out_folders()

        self.fready = os.path.join( self.logdir, "tasks_process_pubmed.ready")
        self.ready = os.path.exists( self.fready )
        if(self.ready):
            self.logger.info("----------- Process PubMed step skipped since it was already computed -----------")
            self.logger.info("----------- Process PubMed step ended -----------")
        
    def __setup_logdir(self, execdir):
        self.logdir = os.path.join( execdir, "logs" )
        if( not os.path.exists(self.logdir) ):
            os.makedirs( self.logdir )

        logf = os.path.join( self.logdir, "process_pubmed.log" )
        logging.basicConfig( filename=logf, encoding="utf-8", filemode="a", level=logging.INFO, format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S' )
        self.logger = logging.getLogger('process_pubmed')
        self.logger.info("----------- Process PubMed step started -----------")

    def _get_arguments(self):
        parser = argparse.ArgumentParser(description='Process Pubmed')
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
            
            self.outDir = self.config["outpath"]

            self.__setup_logdir( execdir )
        except:
            raise Exception("Mandatory fields not found in config. file")
    
    def _setup_out_folders(self):
        self.out = os.path.join( self.outDir, 'process_pubmed' )
        if( not os.path.isdir( self.out ) ) :
            os.makedirs( self.out )
        
        self.predInDir = os.path.join( self.outDir, 'input_prediction' )
        if( not os.path.isdir( self.predInDir ) ) :
            os.makedirs( self.predInDir )

    def __parse_paper_group(self, fname):
        txt = open(fname, 'r').read()
        tree = ET.ElementTree( ET.fromstring( txt ) )
        root = tree.getroot()
        lines = []
        for pm in root.findall('PubmedArticle'):
            md = pm.findall('MedlineCitation')[0]
            pmid = md.findall('PMID')[0].text
            print('----', pmid)
            article = md.findall('Article')[0]
            ptypes = []
            nodesp = article.findall('PublicationTypeList')
            if( len(nodesp) > 0 ):
                pubtype = nodesp[0]
                for at in pubtype.findall('PublicationType'):
                    ptypes.append(at.text)
            else:
                print('\t',pmid, 'no pub type')
            ptypes = '|'.join(ptypes)
            nodes = article.findall('Abstract')
            if( (len(nodesp) > 0) and (len(nodes) > 0) ):
                abstract = article.findall('Abstract')[0]
                for at in abstract.findall('AbstractText'):
                    lab = ''
                    if( 'Label' in at.attrib ):
                        lab = at.attrib['Label']
                    txt = str(at.text)
                    #try:
                    chex = set( list(re.findall('&#x([a-z0-9]+);', txt)) )
                    for it in chex:
                        try:
                            txt = txt.replace( f"&#x{it};", bytes.fromhex(str(it)).decode('utf-8') )
                        except:
                            pass
                    #except:
                    #    pass
                    lines.append( [ pmid, lab, ptypes, txt ] )
            else:
                print('\t',pmid, 'no abstract')
        
        return lines
    
    def _get_grouped_abstracts(self):
        self.logger.info("[Process PubMed step] Task (Getting abstracts of articles associated to NCT raw info) started -----------")
        
        omap = os.path.join( self.out, 'group_abstract_info_pubmed.tsv' )
        gone = set()
        if( os.path.isfile(omap) ):
            fm = open( omap, 'r' )
            for line in fm:
                l = line.replace('\n','')
                if( (len(l) > 2) and (not l.startswith('pmid') ) ):
                    gone.add( l.split('\t')[0] )
            fm.close()
        else:
            fm = open( omap, 'w' )
            fm.write("pmid\tlabel\tpublication_type\ttext\n")
            fm.close()
    
        inmap = os.path.join( self.out, 'complete_mapping_pubmed.tsv' )
        df = pd.read_csv( inmap, sep='\t')
        allids = set( [ str(id) for id in df['pmid'].unique() ] )
        ids = allids - gone
        interval = 500
        k = 0
        c = 1
        self.logger.info( f"\tIt will obtain {len(ids)}/{len(allids)}" )
        while (k < len(ids) ):
            rids = list(ids)[k:k+interval]
            self.logger.info( f"\tRetrieving and parsing chunk {k}/{len(ids)}" )
            ftmp = os.path.join( self.out, f'gr-{c}_tempFile.xml' )
            fetch = Entrez.efetch(db='pubmed', resetmode='text' ,id = (','.join(rids)), rettype='abstract')
            try:
                with open( ftmp, 'wb') as f:
                    f.write(fetch.read())
                try:
                    lines = self.__parse_paper_group( ftmp )
                    if( len(lines) > 0 ):
                        lines = list( map( lambda x: ("\t".join(x)), lines ))
                        with open( omap, 'a' ) as fm:
                            fm.write( "\n".join(lines)+"\n" )
                except:
                    pass
            except:
                pass
            k += interval
            c+=1

        self.logger.info("[Process PubMed step] Task (Getting abstracts of articles associated to NCT raw info) ended -----------")
            
    def _generate_prediction_inputs(self):
        self.logger.info("[Process PubMed step] Task (Generating input for prediction) started -----------")
        
        inmap = os.path.join( self.out, 'group_abstract_info_pubmed.tsv' )
        df = pd.read_csv( inmap, sep='\t')
        df = df[ ~df['text'].isna() ]
        all_pmids = len( df.pmid.unique())
        pmids_with_ctid = df[ df['text'].str.contains('NCT0') ].pmid.unique()
        df = df[ df.pmid.isin(pmids_with_ctid) ]

        txts = {}
        for i in df.index:
            pmid = df.loc[i,'pmid']
            lab = str(df.loc[i, 'label']).replace(' ', '-').replace('/', '-').replace(',', '-').replace('nan', '')
            fname = f"{pmid}_{lab}.txt"

            text = df.loc[i, 'text']

            txts[fname] = text.replace('"','').replace("'",'')
        
        for fname in tqdm(txts):
            text = txts[fname]
            opath = os.path.join( self.predInDir, fname )
            f = open(opath, 'w')
            f.write(text)
            f.close()
        
        self.logger.info("[Process PubMed step] Task (Generating input for prediction) ended -----------")

    def _mark_as_done(self):
        f = open( self.fready, 'w')
        f.close()

        self.logger.info("----------- Process PubMed ended -----------")
        
    def run(self):
        self._get_grouped_abstracts()
        self._generate_prediction_inputs()
        self._mark_as_done()
        
if( __name__ == "__main__" ):
    i = ProcessPubmed()
    i.run()

