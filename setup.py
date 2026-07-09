'''
the setup.py file is very essential part of pakaging and distributing Python projects 
It is used by setuptools to define the configuration of project 
such as its metadeta, its dependency and more
'''
from setuptools import find_packages,setup
from typing import List

def get_requirements()->List[str]:
    '''
    this fuction will return list of requirements
    '''
    requirement_lst:List[str]=[]
    try:
        with open("requirements.txt","r") as file:
            # read lines from the file
            ## process each line
            lines = file.readlines()
            for line in lines:
                requirement = line.strip()
                if requirement and requirement!= '-e .':
                    requirement_lst.append(requirement)
    except FileNotFoundError:
        print("Requirements.txt file not found")

    return requirement_lst
 
setup(
    name = "NetworkIntrusion",
    version= "0.0.1",
    author="Manthan Jikadara",
    author_email="manthanjikadara@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements()

)
