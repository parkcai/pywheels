import os
import json
from typing import Tuple
from typing import Dict
from typing import Optional
from openai import OpenAI
from threading import Lock
from ..i18n import translate


__all__ = [
    "get_answer_online",
    "ModelManager",
    "load_api_keys",
    "get_answer",
]


def get_answer_online(
    api_key: str,
    base_url: str,
    model: str,
    temperature: Optional[float],
    system_prompt: str,
    prompt: str,
)-> str:

    if base_url:
        client = OpenAI(api_key=api_key, base_url=base_url)
        
    else:
        client = OpenAI(api_key=api_key)
        
    if temperature is not None:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature = temperature,
            stream = False
        )
        
    else:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            stream = False
        )
        
    if isinstance(response, str):
        return response
    
    else:
        response_content = response.choices[0].message.content
        
        if response_content is not None:
            return response_content
        
        else:
            return ""


class ModelManager:
    
    # ----------------------------- Model Manager 初始化 ----------------------------- 
    
    def __init__(self):
        
        self._is_online_model: Dict[str, bool] = {}
        
        self._online_models = {}
        self._online_models_lock: Lock = Lock()
        
    # ----------------------------- 外部动作 ----------------------------- 

    def load_api_keys(
        self, 
        api_keys_path: str,
    )-> None:
        
        if not os.path.exists(api_keys_path) or not os.path.isfile(api_keys_path):
                
            raise ValueError(
                translate(" api keys 文件 %s 不存在或不是一个文件！")
                % (api_keys_path)
            )
            
        with open(
            file = api_keys_path, 
            mode = 'r',
            encoding = 'UTF-8',
        ) as file:
                
            api_keys_dict = json.load(file)
        
        with self._online_models_lock:
                
            for model_name in api_keys_dict:
            
                self._is_online_model[model_name] = True
                
                self._online_models[model_name] = {
                    "instances": [
                        {
                            "api_key": api_keys_dict[model_name][index]["api_key"],
                            "base_url": api_keys_dict[model_name][index]["base_url"],
                            "model": api_keys_dict[model_name][index]["model"],
                        }
                        for index in range(len(api_keys_dict[model_name]))
                    ],
                    "next_choice_index": 0,
                }
        
        
    def get_answer(
        self,
        model_name: str, 
        prompt: str,
        model_temperature: Optional[float] = None,
        system_prompt: str = "You are a helpful assistant",
    ):
        
        if not self._is_online_model[model_name]:
            
            raise ValueError(
                translate("模型 %s 未被记录！") % (model_name)
            )
            
        api_key, base_url, model = self._get_online_model_instance(model_name)
        
        llm_answer =  get_answer_online(
            api_key = api_key,
            base_url = base_url,
            model = model,
            temperature = model_temperature,
            system_prompt = system_prompt,
            prompt = prompt,
        )
        
        return llm_answer
                
    # ----------------------------- 内部动作 ----------------------------- 
  
    def _get_online_model_instance(
        self,
        model_name: str,
    )-> Tuple[str, str, str]:
        
        with self._online_models_lock:
            
            online_model = self._online_models[model_name]
            
            index_backup = online_model["next_choice_index"]
            self._online_models[model_name]["next_choice_index"] = \
                (online_model["next_choice_index"]+1) % len(online_model["instances"])
            
            return (
                online_model["instances"][index_backup]["api_key"],
                online_model["instances"][index_backup]["base_url"],
                online_model["instances"][index_backup]["model"],
            )
    
# ----------------------------- 常用 API -----------------------------

model_manager = ModelManager()

def load_api_keys(
    api_keys_path: str,
)-> None:
    
    model_manager.load_api_keys(api_keys_path)
     
        
def get_answer(
    model_name: str, 
    prompt: str,
    model_temperature: Optional[float] = None,
    system_prompt: str = "You are a helpful assistant",
)-> str:
        
    llm_answer = model_manager.get_answer(
        model_name = model_name,
        prompt = prompt,
        model_temperature = model_temperature,
        system_prompt = system_prompt,
    )
    
    return llm_answer


default_api_keys_path = "api_keys.json"

try:
    load_api_keys(default_api_keys_path)
    
except Exception as error:
    pass
