from setuptools import find_packages, setup
from typing import List
HYPHEN_E="-e ."


def get_requirements(file_path:str)->List[str]:
    '''
    This fuction will return list of requirements
    '''
    
    with open(file_path) as f:
        requirements=[]
        requirements=f.readlines()
        requirements=[req.replace('\n','') for req in requirements]

        if HYPHEN_E in requirements:
            requirements.remove(HYPHEN_E)
    return requirements


setup(
    name='mlproject',
    version='0.0.1',
    author='Jisha',
    author_email='jishjuan@gamil.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)