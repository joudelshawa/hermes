from Agents.LLMAgent import Agent
import re
import os
from Utils.Helpers import *
from Utils.Logger import TheLogger, Level
from pydantic import BaseModel

class ReportCreator(Agent):
    def __init__(
            self, 
            base_llm = "deepseek-r1:14b", 
            name = "", 
            system_prompt = "", 
            stream = False,
            max_iter = 3,
            temperature:int = 0.3, 
            top_p:int = 0.4,
            osl_userPrompt:str = "",
            osl_assistantResponse:str = "",
            contextLengthMultiplier:int = 8,
            logger:TheLogger = None
    ):
        oneShotLearningExample = []
        if osl_userPrompt != "" and osl_assistantResponse != "":
            oneShotLearningExample = [
                {
                    "role": "user",
                    "content": osl_userPrompt
                },
                {
                    "role": "assistant",
                    "content": osl_assistantResponse
                }
            ]
        else: 
            logger.log(Level.INFO, 0, f"{name}: Not Using One-Shot-Learning", addTimeTab=False)
        super().__init__(base_llm, name, system_prompt, stream, max_iter, temperature, top_p, oneShotLearningExample, contextLengthMultiplier, logger)
        self.main_headings = [
            "patient",
            "complaint",
            "history",
            "discharge",
            "medications"
        ]
        self.headings1 = ["Patient Report"]
        self.headings2 = ["Comprehensive Patient Profile (CPP)",
                    "Subjective Objective Assessment Plan (SOAP)",
                    "Discharge Conditions (DC)"]
        self.headings3 = ["Administrative & Identification Details",
                    "Subjective",
                    "Objective",
                    "Assessment",
                    "Plan",
                    "Physical Exam at Discharge",
                    "Discharge Condition",
                    "Discharge Instructions",
                    "Disposition",
                    "Brief Hospital Course"]
        self.headings4 = ["Chief Complaint (CC)",
                    "History of Present Illness (HPI)",
                    "Past Medical History (PMH)",
                    "Past Surgical History (PSH)",
                    "Family History (FH)",
                    "Social History (SH)",
                    "Review of Systems (ROS)",
                    "Current Medications and Allergies",
                    "Vital Signs (During Admission)",
                    "Physical Examination",
                    "Laboratory Data",
                    "Imaging and Other Diagnostic Data",
                    "Other Clinicians",
                    "Problem List",
                    "Differential Diagnoses",
                    # "Problem 1",
                    # "Problem 2"
                    ]
        
    
    def extract_headings(self, text):
        headings = {
            1: re.findall(r'^# (.+)', text, re.MULTILINE),
            2: re.findall(r'^## (.+)', text, re.MULTILINE),
            3: re.findall(r'^### (.+)', text, re.MULTILINE),
            4: re.findall(r'^#### (.+)', text, re.MULTILINE),
        }
        return headings

    def check_headings(self, headings, expected_headings):
        missing_headings = []
        for expected in expected_headings:
            exists = False
            for heading in headings:
                if expected in heading:
                    exists = True
                    break
            if not exists:
                missing_headings.append(expected)
        return (False, missing_headings) if len(missing_headings) > 0 else (True, [])
    
    def validateResponse(self, response, **kwargs):
        result = {"is_valid": True, "response": response, "errors": ""}
        headings = self.extract_headings(response)
        check1, missing_1 = self.check_headings(headings[1], self.headings1)
        check2, missing_2 = self.check_headings(headings[2], self.headings2)
        check3, missing_3 = self.check_headings(headings[3], self.headings3)
        check4, missing_4 = self.check_headings(headings[4], self.headings4)
        
        if not check1 or not check2 or not check3 or not check4:
            missing_headings = missing_1 + missing_2 + missing_3 + missing_4
            result["is_valid"] = False
            result["errors"] = f"You must include the following headings: {missing_headings}"
            self.logger.log(Level.ERROR, 1, f"Missing Headings: {missing_headings}")
        return result
    

    # def validateResponse(self, response, unstructured_report:str):
    #     # Add logic for response validation
    #     # look at super class for information on input output
    #     # pattern = r"```output(.*)```"
    #     # match = re.search(pattern, response)
    #     result = {"is_valid": True, "response": response, "errors": ""}
    #     # if match:
    #     #     result["is_valid"] = True
    #     #     result["response"] = match.group(1).strip()
    #     # else:
    #     #     result["errors"].append("Invalid Response - ```output\n...\n``` pattern not found")
    #     #     return result
        
    #     found_headings = re.findall(r"\*\*(.*?)\*\*", result["response"])
    #     found_headings = [
    #         word.lower()
    #         for heading in found_headings
    #         for word in re.sub(r'[^a-zA-Z ]', '', heading).split()
    #     ]

    #     missing_headings = []
    #     # Check if any major heading word is missing
    #     for heading in self.main_headings:
    #         if not any(word in found_headings for word in heading.lower().split()):
    #             missing_headings.append(heading)
    #             result["is_valid"] = False

    #     if not result["is_valid"]:
    #         result["errors"] += f"You must include these words in your headings: {', '.join(missing_headings)}"

    #     # if not areNumbericallyEquivalent(response, unstructured_report):
    #     #     result["is_valid"] = False
    #     #     halucinated, missingNumbers = getMissingNumbers(text_og=unstructured_report, text_gen=response)
    #     #     result["errors"] += f"Missing numerical values (value -> count): {missingNumbers}."
    #     return result
    
    def run(self, prompt, context = ""):
        # Response generation step
        max_iter = self.MAX_ITERATIONS
        tempFolder = os.path.join(self.logger.mainSaveFolder, "Temp/")
        os.makedirs(tempFolder, exist_ok=True)

        while(max_iter > 0):
            self.logger.log(Level.INFO, 1, "|")
            self.logger.log(Level.INFO, 1, f"|\tIteration [{self.MAX_ITERATIONS-max_iter+1}/{self.MAX_ITERATIONS}]", addTimePrefix=True)
            response = remove_think(super().run(prompt, context))
            validation = self.validateResponse(response, unstructured_report=prompt)
            # Temporary Save
            saveReportAsText(validation["response"], tempFolder)
            
            if validation["is_valid"]:
                self.logger.log(Level.SUCCESS, 1,"|\t|---> Success!!")
                return validation["response"]
            else:
                self.logger.log(Level.ERROR, 1,"|\t|---> ERROR!!")
                self.logger.log(Level.ERROR, 1,f"|\t|---> {validation['errors']}", onlyLocalWrite=True)
                if max_iter-1 != 0:
                    self.logger.log(Level.ERROR, 1,"|\t|---> Trying again...")
                    context = f"Your Previous Response: \"\"\"{validation['response']}\"\"\"\n## NOTE\nThe following errors were made in your previous response: \n{validation['errors']}\n"
            max_iter -= 1
        
        self.logger.log(Level.CRITICAL, 0,"="*50)
        self.logger.log(Level.CRITICAL, 1,"HERMES-R FAILED (T_T)!")
        self.logger.log(Level.CRITICAL, 0,"="*50)
        exit()