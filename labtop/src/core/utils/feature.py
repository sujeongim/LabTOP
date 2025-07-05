from abc import ABC

class EHRBase(ABC):
    def __init__(self, cfg):
        pass

class MIMICIV(EHRBase):
    def __init__(self, cfg):
        try:
            self.raw_data_path = cfg.raw_data_path
        except:
            try:
                self.raw_data_path = cfg.data.raw_data_path
            except:
                self.raw_data_path = None
        self.ext = cfg.data.ext

        self.icustay_fname = "icu/icustays" + self.ext
        self.patient_fname = "hosp/patients" + self.ext
        self.admission_fname = "hosp/admissions" + self.ext
        self.diagnosis_fname = "hosp/diagnoses_icd" + self.ext

        self.stayid_key = 'stay_id'
        self.patientid_key = 'subject_id'
        
        self.lab_table= 'labevents'
                
        self.table_candidates = {
            'labevents' : {
                            "fname": "hosp/labevents" + self.ext,
                            "timestamp": "charttime",
                            "use": [ "itemid", "value", "valuenum", "valueuom"], # comments
                            "exclude": [
                                "labevent_id",
                                "storetime",
                                "subject_id",
                                "specimen_id",
                                "order_provider_id", #MIMIC-IV-2.2V added this column
                            ],
                            "code": "itemid",
                            "item_col" : 'itemid',
                            "num_cols" : ["value", "valuenum"],
                            "desc": "hosp/d_labitems" + self.ext,
                            "desc_key": "label",
                            #'itemid' : [50809, 50811, 50912, 50931, 51006, 51221, 51222, 51249]
                            "itemid": [50809, 50811, 50912, 50931, 51006, 51221, 51222, 51249, 51250, 51265, \
                                       51279, 51301, 51478, 51480, 51638, 51639, 51640, 51691, 51755, 51756, \
                                      51981, 52028, 52546, 52569, 52647, 53182, 53189]
                                                                },
            'microbiologyevents': { 
                        "fname": "hosp/microbiologyevents" + self.ext,
                        "timestamp": "charttime",
                        "item_col" : "test_name",
                        "use" : ["spec_type_desc", "test_name", #"test_seq",  "isolate_num",
                                 "org_name", "quantity", "ab_name",
                                 "dilution_text","interpretation", "comments"],
                        "exclude": [
                            'microevent_id',
                            'subject_id',
                            'hadm_id',
                            'micro_specimen_id',
                            'order_provider_id',
                            'chartdate',
                            'spec_itemid',
                            'test_seq',
                            'storedate',
                            'storetime',
                            'test_itemid',
                            'org_itemid',
                            'isolate_num',
                            'ab_itemid',
                        ],
                    },
            'outputevents': {
                        "fname": "icu/outputevents" + self.ext,
                        "timestamp": "charttime",
                        "use": ["itemid","value","valueuom"],
                        "exclude": [
                            'subject_id',
                            'hadm_id',
                            'caregiver_id',
                            'storetime',
                        ],
                        "code": "itemid",
                        "item_col" : 'itemid',
                        "desc": "icu/d_items" + self.ext,
                        "num_cols" : [ "value" ],
                        "desc_key": "label",
                        
                    },
            'inputevents' : {
                        "fname": "icu/inputevents" + self.ext,
                        "timestamp": "starttime",
                        "use": ["itemid", "amount","amountuom","ordercategoryname"],
                        "exclude": [ # only for genhpf
                                "endtime",
                                "storetime",
                                "orderid",
                                "linkorderid",
                                "subject_id",
                                "continueinnextdept",
                                "statusdescription",
                            ],
                        "code": "itemid",
                        "item_col" : 'itemid',
                        "desc": "icu/d_items" + self.ext,
                        "num_cols" : [
                            "amount", 
                        ],
                        "desc_key": "label",
                    }, 
            'emar' : {
                        "fname": "hosp/emar" + self.ext,
                        "timestamp": "charttime",
                        "use": ["medication","dose_given","dose_given_unit","event_txt"],
                        "exclude": [
                            'subject_id',
                            'hadm_id',
                            'poe_id',
                            'pharmacy_id',
                            'enter_provider_id',
                            'scheduletime',
                            'storetime',
                        ],  
                        "item_col" : "medication",
                        "detail" : "hosp/emar_detail" + self.ext,
                        "num_cols" : ["dose_given"],
                        "desc_key": "medication"
                    },
            'procedureevents': {
                        "fname": "icu/procedureevents" + self.ext,
                        "timestamp": "starttime",
                        "use": ["itemid","value","valueuom"],
                        "exclude": [
                            'subject_id',
                            'hadm_id',
                            'caregiver_id',
                            'endtime',
                            'storetime',
                            'orderid',
                            'linkorderid',
                        ],
                        "code": "itemid",
                        "item_col" : "itemid",
                        "desc": "icu/d_items" + self.ext,
                        "num_cols" : ["value"],
                        "desc_key": "label",
                    }, 
        }

            


class eICU(EHRBase):
    def __init__(self, cfg):
        try:
            self.raw_data_path = cfg.raw_data_path
        except:
            self.raw_data_path = cfg.data.raw_data_path
        self.ext = cfg.data.ext

        self.icustay_fname = "patient" + self.ext
        self.patient_fname = "patient" + self.ext
        #self.admission_fname = "hosp/admissions" + self.ext
        #self.diagnosis_fname = "hosp/diagnoses_icd" + self.ext
        self.stayid_key = 'patientunitstayid'
        self.patientid_key = 'uniquepid'
        
        self.lab_table= 'lab'

        self.table_candidates = {
            'lab' : {
                            "fname": "lab" + self.ext,
                            "timestamp": "labresultoffset",
                            "use": [ "labname", "labresult", "labmeasurenamesystem", "labmeasurenameinterface"], # comments
                            "item_col" : 'labname',
                            "num_cols" : ["labresult"],
                    },
            'microLab': { # TODO
                        "fname": "microLab" + self.ext,
                        "timestamp": "culturetakenoffset",
                        "item_col" : "antibiotic",
                        "use" : ["culturesite", "organism", 
                                 "antibiotic", "sensitivitylevel"],
                    },
            'intakeOutput': { # TODO
                        "fname": "intakeOutput" + self.ext,
                        "timestamp": "intakeoutputoffset",
                        "use": ["celllabel", "cellvaluenumeric"],
                        "item_col" : "celllabel",
                        "num_cols" : ["cellvaluenumeric"],
                    },
            'infusionDrug' : {
                        "fname": "infusionDrug" + self.ext,
                        "timestamp": "infusionoffset",
                        "use": ["drugname", "drugamount"],
                        "item_col" : "drugname",
                        "num_cols" : ["drugamount"], 

                    }, 
            'medication' : {
                        "fname": "medication" + self.ext,
                        "timestamp": "drugstartoffset",
                        "use": ["drugname", "dosage"],
                        "item_col" : "drugname",
                        "num_cols" : ["dosage"],
                    },
            'treatment': {
                        "fname": "treatment" + self.ext,
                        "timestamp": "treatmentoffset",
                        "item_col" : "treatmentstring",
                        "use": ["treatmentstring"],
                    }, 
        }
                                      

        

class HIRID(EHRBase):
    def __init__(self, cfg):
        try:
            self.raw_data_path = cfg.raw_data_path
        except:
            self.raw_data_path = cfg.data.raw_data_path
        self.ext = "/parquet"
        
        self.icustay_fname = "general_table.csv" 
        self._ref_fname = "hirid_variable_reference.csv" 
        #self.data_dir = os.path.join(self.cache_dir)
        
        self.stayid_key = 'patientid'
        self.patientid_key = 'patientid'
        
        self.lab_table= 'observation_tables'

        self.table_candidates = {
            'observation_tables' : {
                "fname": "observation_tables" + self.ext,
                "timestamp": "datetime",
                "timeoffsetunit": "abs",
                "use": ["variableid", "value", "Unit"],
                "code": "variableid",
                'num_cols' : ['value'],
                "item_col": "variableid",
                "desc": self._ref_fname,
                "desc_code_col": "ID",
                "desc_key": ["Variable Name", "Unit"],
                "desc_filter_col": "Source Table",
                "desc_filter_val": "Observation",
                "rename_map": [{"Variable Name": "variableid"}],
            },
            'pharma_records' : {
                "fname": "pharma_records" + self.ext,
                "timestamp": "givenat",
                "timeoffsetunit": "abs",
                "use": [
                    "pharmaid",
                    "givendose",
                    "doseunit",
                ],
                "code": "pharmaid",
                "item_col": "pharmaid",
                "desc": self._ref_fname,
                "desc_code_col": "ID",
                "desc_key": ["Variable Name", "Unit"],
                'num_cols' :['givendose'],
                "desc_filter_col": "Source Table",
                "desc_filter_val": "Pharma",
                "rename_map": [{"Variable Name": "variableid"}],
            },
        }