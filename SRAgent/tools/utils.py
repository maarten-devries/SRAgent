# import
## batteries
import json
import shutil
import tempfile
from subprocess import Popen, PIPE
from typing import Annotated, List, Dict, Tuple, Optional, Union, Any
import xml.etree.ElementTree as ET
from xml.parsers.expat import ExpatError
## 3rd party
import xmltodict

# functions
def batch_ids(ids: List[str], batch_size: int) -> List[List[str]]:
    """
    Batch a list of IDs into smaller lists of a given size.
    Args:
        ids: List of IDs.
        batch_size: Size of each batch.
    Returns:
        List of batches.
    """
    for i in range(0, len(ids), batch_size):
        yield ids[i:i + batch_size]

def truncate_values(record, max_length: int) -> str:
    """
    Truncate long values in the record.
    Args:
        record: XML record to truncate.
        max_length: Maximum length of the value.
    Returns:
        Truncated record.
    """
    try:
        root = ET.fromstring(record)
    except ET.ParseError:
        return record
    for item in root.findall(".//Item"):
        if item.text and len(item.text) > max_length:
            item.text = item.text[:max_length] + "...[truncated]"
    # convert back to string
    return ET.tostring(root, encoding="unicode")

def xml2json(record: str) -> Dict[str, Any]:
    """
    Convert an XML record to a JSON object.
    Args:
        record: XML record.
    Returns:
        JSON object.
    """
    try:
        return json.dumps(xmltodict.parse(record), indent=2)
    except ExpatError:
        return record

def run_cmd(cmd: list) -> Tuple[int, str, str]:
    """
    Run sub-command and return returncode, output, and error.
    Args:
        cmd: Command to run
    Returns:
        tuple: (returncode, output, error)
    """
    cmd = [str(i) for i in cmd]
    p = Popen(cmd, stdout=PIPE, stderr=PIPE)
    output, err = p.communicate()
    return p.returncode, output, err