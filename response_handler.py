"""
Response Handler for API responses
Detects and processes structured data, CSV, Excel formats from API responses.
"""

import re
import pandas as pd
import io
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

class ResponseHandler:
    """Handles API responses and detects structured data formats."""
    
    def __init__(self):
        self.output_dir = Path("temp")
        self.output_dir.mkdir(exist_ok=True)
    
    def process_response(self, response_text: str, query: str) -> dict:
        """Process API response and detect structured data formats."""
        result = {
            'text': response_text,
            'has_structured_data': False,
            'files_created': [],
            'data_formats': []
        }
        
        # Detect CSV data in response
        csv_data = self._extract_csv_data(response_text)
        if csv_data:
            excel_file = self._csv_to_excel(csv_data, query)
            if excel_file:
                result['files_created'].append(excel_file)
                result['data_formats'].append('CSV → Excel')
                result['has_structured_data'] = True
        
        # Detect table data
        table_data = self._extract_table_data(response_text)
        if table_data:
            excel_file = self._table_to_excel(table_data, query)
            if excel_file:
                result['files_created'].append(excel_file)
                result['data_formats'].append('Table → Excel')
                result['has_structured_data'] = True
        
        # Detect financial calculations
        financial_data = self._extract_financial_calculations(response_text)
        if financial_data:
            excel_file = self._financial_to_excel(financial_data, query)
            if excel_file:
                result['files_created'].append(excel_file)
                result['data_formats'].append('Financial → Excel')
                result['has_structured_data'] = True
        
        return result
    
    def _extract_csv_data(self, text: str) -> str:
        """Extract CSV-formatted data from response text."""
        # Look for CSV patterns (comma-separated values with headers)
        csv_pattern = r'(?:^|\n)([A-Za-z][^,\n]*(?:,[^,\n]*)+(?:\n[^,\n]*(?:,[^,\n]*)+)*)'
        matches = re.findall(csv_pattern, text, re.MULTILINE)
        
        for match in matches:
            lines = match.split('\n')
            if len(lines) >= 2 and ',' in lines[0]:  # Has header and data
                return match
        
        return None
    
    def _extract_table_data(self, text: str) -> list:
        """Extract table data from pipe-delimited or formatted tables."""
        # Look for pipe-delimited tables
        table_pattern = r'\|[^\n]+\|\n\|[^\n]+\|\n(?:\|[^\n]+\|\n)+'
        matches = re.findall(table_pattern, text)
        
        tables = []
        for match in matches:
            lines = match.strip().split('\n')
            if len(lines) >= 3:  # Header, separator, data
                table_data = []
                for line in lines:
                    if '|' in line and not line.strip().startswith('|--'):
                        cells = [cell.strip() for cell in line.split('|')[1:-1]]
                        if cells and any(cell for cell in cells):
                            table_data.append(cells)
                
                if len(table_data) >= 2:  # Has header and data
                    tables.append(table_data)
        
        return tables
    
    def _extract_financial_calculations(self, text: str) -> dict:
        """Extract financial calculations and formulas."""
        financial_data = {}
        
        # Look for financial formulas and calculations
        formula_patterns = {
            'NPV': r'NPV\s*=\s*([^.\n]+)',
            'IRR': r'IRR\s*=\s*([^.\n]+)',
            'DCF': r'DCF\s*=\s*([^.\n]+)',
            'ROI': r'ROI\s*=\s*([^.\n]+)',
            'Cap Rate': r'Cap\s+Rate\s*=\s*([^.\n]+)',
        }
        
        for metric, pattern in formula_patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                financial_data[metric] = match.group(1).strip()
        
        # Look for numerical data with currency
        currency_pattern = r'\$[\d,]+\.?\d*'
        currency_matches = re.findall(currency_pattern, text)
        if currency_matches:
            financial_data['Currency_Values'] = currency_matches
        
        return financial_data if financial_data else None
    
    def _csv_to_excel(self, csv_data: str, query: str) -> str:
        """Convert CSV data to Excel file."""
        try:
            # Parse CSV data
            df = pd.read_csv(io.StringIO(csv_data))
            
            # Create Excel file
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"API_Response_CSV_{timestamp}.xlsx"
            filepath = self.output_dir / filename
            
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Data', index=False)
                
                # Add metadata sheet
                metadata = pd.DataFrame({
                    'Field': ['Query', 'Generated', 'Source', 'Format'],
                    'Value': [query, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'API Response', 'CSV']
                })
                metadata.to_excel(writer, sheet_name='Metadata', index=False)
            
            logger.info(f"CSV converted to Excel: {filename}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error converting CSV to Excel: {e}")
            return None
    
    def _table_to_excel(self, table_data: list, query: str) -> str:
        """Convert table data to Excel file."""
        try:
            for i, table in enumerate(table_data):
                # Create DataFrame from table
                df = pd.DataFrame(table[1:], columns=table[0])
                
                # Create Excel file
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"API_Response_Table_{timestamp}.xlsx"
                filepath = self.output_dir / filename
                
                with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                    df.to_excel(writer, sheet_name=f'Table_{i+1}', index=False)
                    
                    # Add metadata
                    metadata = pd.DataFrame({
                        'Field': ['Query', 'Generated', 'Source', 'Format'],
                        'Value': [query, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'API Response', 'Table']
                    })
                    metadata.to_excel(writer, sheet_name='Metadata', index=False)
                
                logger.info(f"Table converted to Excel: {filename}")
                return str(filepath)
            
        except Exception as e:
            logger.error(f"Error converting table to Excel: {e}")
            return None
    
    def _financial_to_excel(self, financial_data: dict, query: str) -> str:
        """Convert financial calculations to Excel file."""
        try:
            # Create financial analysis sheet
            analysis_data = []
            for metric, value in financial_data.items():
                if metric != 'Currency_Values':
                    analysis_data.append({'Metric': metric, 'Formula/Value': value})
            
            # Create Excel file
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"API_Response_Financial_{timestamp}.xlsx"
            filepath = self.output_dir / filename
            
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                if analysis_data:
                    df = pd.DataFrame(analysis_data)
                    df.to_excel(writer, sheet_name='Financial Analysis', index=False)
                
                # Currency values sheet
                if 'Currency_Values' in financial_data:
                    currency_df = pd.DataFrame({
                        'Currency Values': financial_data['Currency_Values']
                    })
                    currency_df.to_excel(writer, sheet_name='Currency Values', index=False)
                
                # Metadata
                metadata = pd.DataFrame({
                    'Field': ['Query', 'Generated', 'Source', 'Format'],
                    'Value': [query, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'API Response', 'Financial']
                })
                metadata.to_excel(writer, sheet_name='Metadata', index=False)
            
            logger.info(f"Financial data converted to Excel: {filename}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error converting financial data to Excel: {e}")
            return None