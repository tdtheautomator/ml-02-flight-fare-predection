from setuptools import find_packages,setup
from typing import List

#populate list from list from requirements.txt 
starter ='-e .'
def get_requirements(file_path:str)->List[str]:
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]
        #remove -e from the requirements.txt
        if starter in requirements:
            requirements.remove(starter)
    return requirements

#metadat for the project
setup(
name='ml-02-flight-fare-predection',
version='0.1.0',
author='tdtheautomator',
packages=find_packages(),
install_requires=get_requirements('requirements.txt')
)