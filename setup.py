from setuptools import setup, find_packages
from typing import List

def get_requirements(filepath: str) -> List[str]:
    
    with open(filepath) as file_obj:
        requirements=file_obj.read().splitlines()
        requirements=[req.replace("\n","") for req in requirements]
        requirements = [req.replace("==", ">=") for req in requirements]
        if '-e .' in requirements:
            requirements.remove('-e .')
    return requirements

setup(
    name='cnn_drug_discovery',
    version='0.1',
    author='Abhi5ingh',
    author_email="abhisteak@gmail.com",
    description='CNN Drug Discovery',
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
)