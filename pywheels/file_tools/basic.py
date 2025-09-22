from ..i18n import translate
from ..typing import *
from ..external import *


__all__ = [
    "guarantee_file_exist",
    "assert_file_exist",
    "append_to_file",
    "get_temp_file_path",
    "delete_file",
    "copy_file",
    "clear_file",
    "get_file_paths",
    "get_lines",
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
    immediate_flush: bool = True,
    buffering: Optional[int] = None,
):

    with open(
        file = file_path, 
        mode = "a", 
        encoding = encoding,
        buffering = -1 if buffering is None else buffering, 
    ) as file_pointer:
        
        file_pointer.write(content + end)
        if immediate_flush: file_pointer.flush()
        
        
tempfile_lock = Lock()

def get_temp_file_path(
    suffix: Optional[str] = "",
    prefix: str = "tmp_",
    directory: Optional[str] = None,
)-> str:

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
    source_path: str,
    destination_path: str,
)-> None:

    if not os.path.exists(source_path):
        raise FileNotFoundError(
            translate(
                "[copy_file 报错] 源路径不存在: %s"
            ) % (source_path)
        )
    
    destination_parent_directory = os.path.dirname(destination_path)
    if destination_parent_directory:
        os.makedirs(destination_parent_directory, exist_ok=True)

    if os.path.isfile(source_path):
        shutil.copy2(source_path, destination_path)
    elif os.path.isdir(source_path):
        shutil.copytree(source_path, destination_path, dirs_exist_ok=True)
    else:
        raise ValueError(
            translate(
                "[copy_file 报错] 不支持的源路径类型: %s"
            ) % (source_path)
        )

        
def clear_file(
    file_path: str,
    encoding: str = "UTF-16",
):

    try:
        with open(
            file = file_path, 
            mode = "w", 
            encoding = encoding,
        ):
            pass
    except FileNotFoundError:
        guarantee_file_exist(file_path)
        
        
def get_file_paths(
    directory: str,
    file_type: Literal["all", "files_only", "dirs_only"] = "all",
    starting_with: str = "",
    ending_with: str = "",
    return_format: Literal["name_only", "full_path"] = "name_only",
    sort_key: Callable[[str], float] = lambda _: 0.0,
    sort_reverse: bool = False,
)-> List[str]:
    
    if not os.path.exists(directory):
        raise ValueError(
            translate("[get_file_paths 报错] 目录 %s 不存在")
            % (directory)
        )
    
    if not os.path.isdir(directory):
        raise ValueError(
            translate("[get_file_paths 报错] 路径 %s 不是目录")
            % (directory)
        )
    
    result = []
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if not (item.startswith(starting_with) and item.endswith(ending_with)): continue
        if file_type == "all":
            result.append(item)
        elif file_type == "dirs_only" and os.path.isdir(item_path):
            result.append(item)
        elif file_type == "files_only" and os.path.isfile(item_path):
            result.append(item)
            
    result.sort(key=sort_key, reverse=sort_reverse)
            
    if return_format == "name_only":
        return result
    elif return_format == "full_path":
        result = [f"{directory}/{name}" for name in result]
        return result
    else:
        raise ValueError(
            translate(
                "[get_file_paths 出错] 未知的 return_format 取值：%s！"
            ) % (return_format)
        )


def get_lines(
    file_path: str,
    strip: bool = True,
    encoding: str = "UTF-8",
)-> List[str]:
    
    with open(
        file = file_path,
        mode = "r",
        encoding = encoding,
    ) as file_pointer:
        
        lines = file_pointer.readlines()
        
    if strip:
        lines = [line.strip() for line in lines]
        
    return lines