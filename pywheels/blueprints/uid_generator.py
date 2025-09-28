from typing import Set
from typing import Optional
from threading import Lock
from random import Random
from string import ascii_lowercase


__all__ = [
    "UIDGenerator",
]


class UIDGenerator:
    
    def __init__(
        self,
        seed: Optional[int] = None,
    )-> None:
        
        self._lock = Lock()
        self._recorded_uids: Set[str] = set()
        self._random_generator: Random = Random(seed)
        
        
    def generate(
        self,
        uid_length: int,
    )-> str:
        
        with self._lock:
            while True:
                trial_uid: str = self._get_trial_uid(uid_length)
                if trial_uid not in self._recorded_uids:
                    self._recorded_uids.add(trial_uid)
                    return trial_uid
                
                
    def update_existing_uids(
        self,
        existing_uids: Set[str],
    )-> None:
        
        with self._lock:
            self._recorded_uids.update(existing_uids)
            

    def _get_trial_uid(
        self,
        uid_length: int,
    )-> str:
        
        trial_uid: str = ''.join(self._random_generator.choices(
            population = ascii_lowercase, 
            k = uid_length,
        ))
        return trial_uid