import json 

class LLMLogExtractor():
    
    def __init__(self, log_fp):
        self.log_fp = log_fp
        with open(log_fp, 'r') as log_fp:
            log_raw = log_fp.readlines()
            log = ''.join(log_raw)[1:-1].split('}{')
        self.log = None
        self.log = [LLMLogExtractor.format(log_item) for log_item in log]
        
    @staticmethod
    def format(log_item):
        log_item = '{'+log_item+'}'
        log_item = log_item.replace('\n','').replace(' ','')
        log_item = log_item.replace('null','"null"')
        return eval(log_item)


    def extract_param(self)->list:
        param_list = []
        for log_item in self.log:
            param_list.append(eval(log_item['choices'][0]['message']['content']))
        return param_list
    
if __name__ == '__main__':
    log_fp = '../../../llmlog/message_12-12-2023-11-34-41.txt'
    llm_log_extractor = LLMLogExtractor(log_fp)
    llm_log_extractor.extract_param()