from string import Template
from typing import List, Dict
from pycuda.compiler import SourceModule


def compile_files(files: List[str], header_files: List[str]=None, template: Dict=None) -> SourceModule:
    source = _get_source_from_files(header_files, template)
    source += _get_source_from_files(files, template)

    module = SourceModule(source)

    return module


def _get_source_from_files(files, template):
    content = ''
    for file in files:
        raw_content = _get_file_content(file)
        content += _apply_template(raw_content, template)

    return content


def _apply_template(content, template):
    return Template(content).substitute(template)


def _get_file_content(path):
    with open(path, 'r') as file:
        content = file.read()

    return content
