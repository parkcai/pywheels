import os
import shutil
import tempfile
from threading import Lock
from typing import Optional
from ..i18n import translate


__all__ = [
    "guarantee_file_exist",
    "assert_file_exist",
    "append_to_file",
    "get_temp_file_path",
    "delete_file",
    "copy_file",
]


def guarantee_file_exist(
    file_path: str,
    is_directory: bool = False,
):
    
    """
    确保给定文件或目录存在，如果不存在则创建它。

    参数:
    file (str): 要检查或创建的文件或目录路径。
    is_directory (bool): 如果为 True，则创建目录而不是文件。
    """
    
    if is_directory:
        os.makedirs(file_path, exist_ok=True)
        
    else:
        
        parent = os.path.dirname(file_path)
        
        if parent:
            os.makedirs(parent, exist_ok=True)
            
        if not os.path.exists(file_path):
            
            with open(
                file = file_path, 
                mode = 'w', 
                encoding = 'UTF-8'
            ):
                pass
        
 
def assert_file_exist(
    file_path: str, 
    error_message = None,
):
    
    """
    断言给定文件存在
    """
    
    if not os.path.exists(file_path):
        
        if error_message is None:
            
            assert False, translate(
                "程序出错，文件 %s 不存在！" % (file_path) 
            )
            
        else:
            assert False, error_message
        

def append_to_file(
    file_path: str, 
    content: str,
    end: str = "\n",
    encoding: str = "UTF-8",
):

    with open(
        file = file_path, 
        mode = "a", 
        encoding = encoding
    ) as file:
        file.write(content + end)
        
        
tempfile_lock = Lock()

def get_temp_file_path(
    suffix: Optional[str] = "",
    prefix: str = "tmp_",
    directory: Optional[str] = None,
) -> str:
    
    """
    在线程安全的环境中生成一个临时文件路径或临时目录。

    本函数会：
      - 如果 suffix 为 None，则创建一个临时目录并返回其路径；
      - 否则，在线程锁保护下生成一个唯一的临时文件路径（不创建文件）。

    Args:
        suffix (str or None): 文件后缀名；若为 None，则表示创建临时目录。
        prefix (str): 文件或目录前缀，默认 "tmp_"。
        directory (str): 保存路径，默认使用临时目录。

    Returns:
        str: 生成的临时文件路径或临时目录路径。
    """

    global tempfile_lock
    
    with tempfile_lock:
        
        if suffix is None:
            
            tmp_dir_path = tempfile.mkdtemp(
                prefix = prefix,
                dir = directory,
            )
            
            return tmp_dir_path
        
        else:
            
            temp_file_path = tempfile.mktemp(
                suffix = suffix,
                prefix = prefix,
                dir = directory,
            )
            
            return temp_file_path
        
        
def delete_file(
    file_path: str
) -> None:
    
    """
    Recursively delete the specified file or directory.

    Under thread lock protection:
      - If it's a file, delete it directly;
      - If it's a directory, recursively delete all its contents and itself;
      - If the path doesn't exist, silently ignore.

    Args:
        file_path (str): Path of the file or directory to be deleted.
    """
    
    global tempfile_lock
    
    with tempfile_lock:
        
        try:
            
            if os.path.isdir(file_path):
                shutil.rmtree(file_path)
                
            elif os.path.isfile(file_path):
                os.remove(file_path)
                
        except FileNotFoundError:
            pass
        
        except Exception as error:
            print(
                translate("删除失败: %s, 错误: %s") % (file_path, error)
            )
            
            
def copy_file(
    source_file_path: str,
    destination_file_path: str,
) -> None:
    
    """
    Copy file or folder from source_file_path to destination_file_path.

    Automatically determines:
      - If source_file_path is a file, copies it as destination_file_path;
      - If source_file_path is a directory:
          - If destination_file_path doesn't exist, recursively copies the entire directory;
          - If destination_file_path exists, copies directory contents inside destination_file_path.
      - Automatically creates parent directories of destination_file_path (if they don't exist)

    Args:
        source_file_path (str): Source path, can be file or directory.
        destination_file_path (str): Destination path.
    """
    
    if not os.path.exists(source_file_path):
        
        raise FileNotFoundError(
            translate("源路径不存在: %s") % (source_file_path)
        )
    
    # Create parent directories of destination_file_path (whether file or directory target)
    os.makedirs(os.path.dirname(destination_file_path), exist_ok=True)

    if os.path.isfile(source_file_path):
        shutil.copy2(source_file_path, destination_file_path)
        
    elif os.path.isdir(source_file_path):
        
        if not os.path.exists(destination_file_path):
            shutil.copytree(source_file_path, destination_file_path)
            
        else:
            
            for item in os.listdir(source_file_path):
                
                s_item = os.path.join(source_file_path, item)
                d_item = os.path.join(destination_file_path, item)
                
                if os.path.isdir(s_item):
                    shutil.copytree(s_item, d_item, dirs_exist_ok=True)
                    
                else:
                    shutil.copy2(s_item, d_item)
                    
    else:
        
        raise ValueError(
            translate("不支持的源路径类型: %s") % (source_file_path)
        )