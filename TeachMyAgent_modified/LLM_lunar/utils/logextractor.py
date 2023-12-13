import json 

class LLMLogExtractor():
    
    def __init__(self, log_fp):
        self.log_fp = log_fp
        with open(log_fp, 'r') as log_fp:
            log_raw = log_fp.readlines()
            log = ''.join(log_raw)[1:-1].split('}{')
        self.log = None
        self.log = [LLMLogExtractor.format(log_item) for log_item in log]
        
    def extract_param(self)->list:
        param_list = []
        for log_item in self.log:
            param_list.append(LLMLogExtractor.extract_dict(log_item['choices'][0]['message']['content']))
        return param_list
    
    @staticmethod
    def format(log_item):
        log_item = '{'+log_item+'}'
        log_item = log_item.replace('\n','').replace(' ','')
        log_item = log_item.replace('null','"null"')
        return eval(log_item)

    @staticmethod
    def extract_dict(content_str):
        extract = content_str.split('{')[-1].split('}')[0].split(',')
        ret_key = [x.split(':')[0] for x in extract]
        ret_value = [float(x.split(':')[1]) for x in extract]
        ret = {}
        for i in range(3):
            ret[eval(ret_key[i])] = ret_value[i]
        return ret

    
if __name__ == '__main__':
    log_fp = '../../../llmlog/message_12-12-2023-14-33-08.txt'
    llm_log_extractor = LLMLogExtractor(log_fp)
    rest = llm_log_extractor.extract_param()
    print(rest)