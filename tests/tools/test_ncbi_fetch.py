import pytest
from unittest.mock import patch, MagicMock
from bs4 import BeautifulSoup
from SRAgent.tools.ncbi_fetch import (
    fetch_geo_record,
    fetch_ncbi_record,
    fetch_pubmed_record,
    _fetch_geo_record,
    _fetch_ncbi_record,
    _fetch_pubmed_record,
    _extract_geo_sections
)

class MockResponse:
    def __init__(self, status_code, text):
        self.status_code = status_code
        self.text = text

@pytest.fixture
def mock_ncbi_response():
    """Return mock HTML for NCBI SRA page"""
    html = """
    <html>
        <div id="maincontent">
            <p class="details expand e-hidden">
                <span>Title:</span> Single-cell RNA-seq analysis
                <span>Organism:</span> Homo sapiens
                <span>Platform:</span> ILLUMINA
            </p>
            <a href="/geo/query/acc.cgi?acc=GSE123456">GSE123456</a>
        </div>
    </html>
    """
    return html

@pytest.fixture
def mock_geo_response():
    """Return mock HTML for GEO page"""
    html = """
    <html>
        <tr>
            <td>Status</td>
            <td>Public on Jul 10, 2023</td>
        </tr>
        <tr>
            <td>Title</td>
            <td>Single-cell RNA-seq of human cells</td>
        </tr>
        <tr>
            <td>Organism</td>
            <td>Homo sapiens</td>
        </tr>
        <tr>
            <td>Platforms</td>
            <td>GPL24676</td>
        </tr>
        <tr>
            <td>Samples</td>
            <td>GSM123456, GSM123457</td>
        </tr>
    </html>
    """
    return html

@pytest.fixture
def mock_pubmed_response():
    """Return mock HTML for PubMed page"""
    html = """
    <html>
        <div class="abstract-content selected">
            <p>This study demonstrates the use of single-cell RNA sequencing to profile human cells.</p>
        </div>
    </html>
    """
    return html

def test_fetch_ncbi_record(mock_ncbi_response):
    """Test fetch_ncbi_record function"""
    with patch('requests.get') as mock_get:
        mock_get.return_value = MockResponse(200, mock_ncbi_response)
        
        result = fetch_ncbi_record.invoke({
            "terms": ["29110018"],
            "database": "sra"
        })
        
        # Verify correct URL was called
        mock_get.assert_called_with("https://www.ncbi.nlm.nih.gov/sra/?term=29110018")
        
        # Verify content
        assert "Query term: 29110018" in result
        assert "Single-cell RNA-seq analysis" in result
        assert "Homo sapiens" in result
        assert "ILLUMINA" in result

def test_fetch_ncbi_record_multiple_terms(mock_ncbi_response):
    """Test fetch_ncbi_record with multiple terms"""
    with patch('requests.get') as mock_get:
        mock_get.return_value = MockResponse(200, mock_ncbi_response)
        
        result = fetch_ncbi_record.invoke({
            "terms": ["29110018", "29110015"],
            "database": "sra"
        })
        
        # Verify both terms were queried
        assert mock_get.call_count == 2
        assert "Query term: 29110018" in result
        assert "Query term: 29110015" in result

def test_fetch_ncbi_record_http_error():
    """Test fetch_ncbi_record with HTTP error"""
    with patch('requests.get') as mock_get:
        mock_get.return_value = MockResponse(404, "Not Found")
        
        result = fetch_ncbi_record.invoke({
            "terms": ["invalid_id"],
            "database": "sra"
        })
        
        assert "Error: Unable to fetch data" in result
        assert "Status code: 404" in result

def test_fetch_ncbi_record_no_content():
    """Test fetch_ncbi_record with no content found"""
    with patch('requests.get') as mock_get:
        # HTML with no matching content
        mock_get.return_value = MockResponse(200, "<html><body>No details found</body></html>")
        
        result = fetch_ncbi_record.invoke({
            "terms": ["29110018"],
            "database": "sra"
        })
        
        assert "Error: Unable to locate details" in result

def test_fetch_geo_record(mock_geo_response):
    """Test fetch_geo_record function"""
    with patch('requests.get') as mock_get:
        mock_get.return_value = MockResponse(200, mock_geo_response)
        
        result = fetch_geo_record.invoke({
            "GEO_accessions": ["GSE123456"]
        })
        
        # Verify correct URL was called
        mock_get.assert_called_with("https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE123456")
        
        # Verify content
        assert "Query term: GSE123456" in result
        assert "Title" in result
        assert "Single-cell RNA-seq of human cells" in result
        assert "Homo sapiens" in result

def test_fetch_geo_record_multiple_accessions(mock_geo_response):
    """Test fetch_geo_record with multiple accessions"""
    with patch('requests.get') as mock_get:
        mock_get.return_value = MockResponse(200, mock_geo_response)
        
        result = fetch_geo_record.invoke({
            "GEO_accessions": ["GSE123456", "GSE789012"]
        })
        
        # Verify both accessions were queried
        assert mock_get.call_count == 2
        assert "Query term: GSE123456" in result
        assert "Query term: GSE789012" in result

def test_fetch_geo_record_http_error():
    """Test fetch_geo_record with HTTP error"""
    with patch('requests.get') as mock_get:
        mock_get.return_value = MockResponse(404, "Not Found")
        
        result = fetch_geo_record.invoke({
            "GEO_accessions": ["invalid_id"]
        })
        
        assert "Error: Unable to fetch data" in result
        assert "Status code: 404" in result

def test_fetch_pubmed_record(mock_pubmed_response):
    """Test fetch_pubmed_record function"""
    with patch('requests.get') as mock_get:
        mock_get.return_value = MockResponse(200, mock_pubmed_response)
        
        result = fetch_pubmed_record.invoke({
            "terms": ["29110018"]
        })
        
        # Verify correct URL was called
        mock_get.assert_called_with("https://pubmed.ncbi.nlm.nih.gov/29110018")
        
        # Verify content
        assert "Query term: 29110018" in result
        assert "single-cell RNA sequencing" in result

def test_fetch_pubmed_record_multiple_terms(mock_pubmed_response):
    """Test fetch_pubmed_record with multiple terms"""
    with patch('requests.get') as mock_get:
        mock_get.return_value = MockResponse(200, mock_pubmed_response)
        
        result = fetch_pubmed_record.invoke({
            "terms": ["29110018", "29110015"]
        })
        
        # Verify both terms were queried
        assert mock_get.call_count == 2
        assert "Query term: 29110018" in result
        assert "Query term: 29110015" in result

def test_fetch_pubmed_record_http_error():
    """Test fetch_pubmed_record with HTTP error"""
    with patch('requests.get') as mock_get:
        mock_get.return_value = MockResponse(404, "Not Found")
        
        result = fetch_pubmed_record.invoke({
            "terms": ["invalid_id"]
        })
        
        assert "Error: Unable to fetch data" in result
        assert "Status code: 404" in result

def test_fetch_pubmed_record_no_content():
    """Test fetch_pubmed_record with no content found"""
    with patch('requests.get') as mock_get:
        # HTML with no matching content
        mock_get.return_value = MockResponse(200, "<html><body>No abstract found</body></html>")
        
        result = fetch_pubmed_record.invoke({
            "terms": ["29110018"]
        })
        
        assert "Error: Unable to locate details" in result

def test_extract_geo_sections(mock_geo_response):
    """Test _extract_geo_sections function"""
    mock_response = MockResponse(200, mock_geo_response)
    
    sections = _extract_geo_sections(mock_response, "GSE123456")
    
    assert len(sections) > 0
    assert any("Title" in section for section in sections)
    assert any("Organism" in section for section in sections)
    assert any("Platforms" in section for section in sections)

def test_extract_geo_sections_no_content():
    """Test _extract_geo_sections with no matching content"""
    mock_response = MockResponse(200, "<html><body>No sections found</body></html>")
    
    sections = _extract_geo_sections(mock_response, "GSE123456")
    
    assert len(sections) == 1
    assert "No data found" in sections[0]