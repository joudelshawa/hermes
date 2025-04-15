import os
from datetime import datetime
from colorama import init, Fore, Style

init(autoreset=True)

class Level:
    HEADING_0 = Fore.GREEN
    HEADING_1 = Fore.LIGHTCYAN_EX
    HEADING_2 = Fore.YELLOW
    INFO = Fore.WHITE
    WARNING = Fore.YELLOW
    ERROR = Fore.RED
    SUCCESS = Fore.GREEN
    CRITICAL = Fore.MAGENTA

class TheLogger:
    
    def __init__(self, modelName, saveToFolder):
        
        if saveToFolder != None and saveToFolder != "":
            self.mainSaveFolder = os.path.join(saveToFolder, modelName)
            logsFolder = os.path.join(self.mainSaveFolder, "Logs")
            os.makedirs(logsFolder, exist_ok=True)
            saveIdx = len(os.listdir(logsFolder)) + 1
            self.localFilePath = os.path.join(logsFolder, f"{saveIdx}_experiment.log")
        else:
            self.mainSaveFolder = None
            self.localFilePath = None
            self._consoleWrite(Level.ERROR, "Not Saving Logs Locally")

        

    def _localWrite(self, msg: str):
        with open(self.localFilePath, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
    
    def _consoleWrite(self, level: str, msg: str) -> None:
        if level == Level.HEADING_0 or level == Level.HEADING_1 or level == Level.HEADING_2 or Level.CRITICAL:
            string = f"{Style.BRIGHT}" + level + msg
        else:
            string = level + msg
        print(string)

    def log(self, level: str, initialTabs: int, msg: str, onlyLocalWrite:bool = False, addTimePrefix=False, addTimeTab=True):
        timestamp = datetime.now().strftime("%H:%M")
        if addTimePrefix:
            prefix = f"[{timestamp}]" 
        else:
            prefix = "       " if addTimeTab else ""
        indented_msg = prefix + "\t" * initialTabs + f"{msg}"

        if not onlyLocalWrite:
            self._consoleWrite(level, indented_msg)
        if self.localFilePath != None: 
            self._localWrite(indented_msg)
