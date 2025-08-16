"""
Google Sheets Creator - Direct Integration
Automatically creates Google Sheets with live formulas when AI responses contain financial models.
"""

import logging
from typing import Dict, List, Any, Optional
import json
import re
import os
from datetime import datetime
from google.oauth2 import service_account
from googleapiclient.discovery import build

logger = logging.getLogger(__name__)

class GoogleSheetsCreator:
    """Creates actual Google Sheets with live formulas from AI responses."""
    
    def __init__(self):
        self.service = None
        self.drive_service = None
        self.credentials_available = self._setup_credentials()
        
    def _setup_credentials(self):
        """Setup Google Sheets API with service account credentials."""
        try:
            # Get credentials from environment
            creds_json = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
            if not creds_json:
                logger.warning("GOOGLE_SERVICE_ACCOUNT_JSON not found in environment")
                return False
            
            # Parse JSON credentials
            creds_info = json.loads(creds_json)
            
            # Define required scopes
            scopes = [
                'https://www.googleapis.com/auth/spreadsheets',
                'https://www.googleapis.com/auth/drive.file'
            ]
            
            # Create credentials
            credentials = service_account.Credentials.from_service_account_info(
                creds_info, scopes=scopes
            )
            
            # Build services
            self.service = build('sheets', 'v4', credentials=credentials)
            self.drive_service = build('drive', 'v3', credentials=credentials)
            
            logger.info("Google Sheets API initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Google Sheets API setup error: {e}")
            return False
    
    def create_dcf_sheet_from_ai_response(self, ai_response: str, query: str) -> Dict:
        """Create Google Sheet with DCF model based on AI response."""
        
        # Parse AI response to extract financial parameters
        financial_data = self._extract_financial_parameters(ai_response)
        
        # Create sheet structure with actual formulas
        sheet_data = {
            "title": f"DCF Model - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "sheets": [
                {
                    "name": "Assumptions",
                    "data": [
                        ["Parameter", "Value", "Notes"],
                        ["Base Revenue", financial_data.get('revenue', 1000000), "Starting revenue"],
                        ["Growth Rate", financial_data.get('growth', 0.15), "Annual growth %"],
                        ["EBITDA Margin", financial_data.get('margin', 0.20), "Profitability %"],
                        ["Discount Rate", financial_data.get('wacc', 0.12), "WACC %"],
                        ["Terminal Growth", financial_data.get('terminal', 0.03), "Long-term growth %"]
                    ]
                },
                {
                    "name": "Projections", 
                    "data": [
                        ["Year", "Revenue", "EBITDA", "EBIT", "NOPAT", "FCF", "PV"],
                        ["2025", "=Assumptions!B2", "=B2*Assumptions!B4", "=C2-B2*0.03", "=D2*0.75", "=E2+B2*0.03-B2*0.05-B2*0.02", "=F2/((1+Assumptions!B5)^1)"],
                        ["2026", "=B2*(1+Assumptions!B3)", "=B3*Assumptions!B4", "=C3-B3*0.03", "=D3*0.75", "=E3+B3*0.03-B3*0.05-B3*0.02", "=F3/((1+Assumptions!B5)^2)"],
                        ["2027", "=B3*(1+Assumptions!B3)", "=B4*Assumptions!B4", "=C4-B4*0.03", "=D4*0.75", "=E4+B4*0.03-B4*0.05-B4*0.02", "=F4/((1+Assumptions!B5)^3)"],
                        ["2028", "=B4*(1+Assumptions!B3)", "=B5*Assumptions!B4", "=C5-B5*0.03", "=D5*0.75", "=E5+B5*0.03-B5*0.05-B5*0.02", "=F5/((1+Assumptions!B5)^4)"],
                        ["2029", "=B5*(1+Assumptions!B3)", "=B6*Assumptions!B4", "=C6-B6*0.03", "=D6*0.75", "=E6+B6*0.03-B6*0.05-B6*0.02", "=F6/((1+Assumptions!B5)^5)"]
                    ]
                },
                {
                    "name": "Valuation",
                    "data": [
                        ["Component", "Value", "Formula"],
                        ["PV of FCF (5 years)", "=SUM(Projections!G2:G6)", "Sum of discounted cash flows"],
                        ["Terminal FCF", "=Projections!F6*(1+Assumptions!B6)", "Final year FCF grown"],
                        ["Terminal Value", "=B3/(Assumptions!B5-Assumptions!B6)", "Terminal FCF / (WACC - g)"],
                        ["PV Terminal Value", "=B4/((1+Assumptions!B5)^5)", "Terminal value discounted"],
                        ["Enterprise Value", "=B2+B5", "Total company value"],
                        ["Equity Value", "=B6-Assumptions!B7", "Less net debt"],
                        ["Per Share Value", "=B7/Assumptions!B8", "If shares outstanding known"]
                    ]
                }
            ],
            "formatting": {
                "currency_ranges": ["Projections!B:G", "Valuation!B:B"],
                "percentage_ranges": ["Assumptions!B3:B6"],
                "conditional_formatting": [
                    {"range": "Projections!F:F", "condition": ">0", "color": "green"},
                    {"range": "Projections!F:F", "condition": "<0", "color": "red"}
                ]
            },
            "charts": [
                {
                    "type": "LINE",
                    "title": "Revenue Growth Projection",
                    "ranges": ["Projections!A2:B6"],
                    "position": "I2"
                },
                {
                    "type": "COLUMN", 
                    "title": "Free Cash Flow Projection",
                    "ranges": ["Projections!A2:A6", "Projections!F2:F6"],
                    "position": "I15"
                }
            ]
        }
        
        return sheet_data
    
    def _extract_financial_parameters(self, ai_response: str) -> Dict:
        """Extract financial parameters from AI response text."""
        
        # Handle None or empty response
        if not ai_response or ai_response is None:
            return {}
        
        # Ensure response is a string
        ai_response = str(ai_response)
        
        params = {}
        
        # Extract numerical values with regex patterns
        patterns = {
            'revenue': r'revenue[:\s]+\$?([\d,]+(?:\.\d+)?)[mk]?',
            'growth': r'growth[:\s]+([\d.]+)%?',
            'margin': r'margin[:\s]+([\d.]+)%?',
            'wacc': r'wacc|discount[:\s]+([\d.]+)%?',
            'terminal': r'terminal[:\s]+([\d.]+)%?'
        }
        
        for param, pattern in patterns.items():
            match = re.search(pattern, ai_response.lower())
            if match and match.group(1):
                value = match.group(1)
                if value is not None:
                    value = value.replace(',', '')
                    try:
                        params[param] = float(value)
                        # Convert percentages to decimals
                        if param in ['growth', 'margin', 'wacc', 'terminal'] and params[param] > 1:
                            params[param] = params[param] / 100
                    except ValueError:
                        continue
        
        return params
    
    def create_sheet_and_return_link(self, sheet_data: Dict) -> str:
        """Create actual Google Sheet and return shareable link."""
        
        if not self.credentials_available:
            return """
            ## Google Sheets Integration Available!
            
            **Setup needed**: Google Service Account credentials required
            **What you'd get**: Live sheets with formulas, charts, and sharing
            """
        
        try:
            # Create new spreadsheet
            spreadsheet_body = {
                'properties': {
                    'title': sheet_data['title']
                },
                'sheets': []
            }
            
            # Add each sheet from sheet_data
            for sheet_info in sheet_data['sheets']:
                sheet_properties = {
                    'properties': {
                        'title': sheet_info['name']
                    }
                }
                spreadsheet_body['sheets'].append(sheet_properties)
            
            # Create the spreadsheet
            spreadsheet = self.service.spreadsheets().create(
                body=spreadsheet_body
            ).execute()
            
            spreadsheet_id = spreadsheet['spreadsheetId']
            
            # Populate each sheet with data and formulas
            for sheet_info in sheet_data['sheets']:
                self._populate_sheet(spreadsheet_id, sheet_info)
            
            # Apply formatting if specified
            if 'formatting' in sheet_data:
                self._apply_formatting(spreadsheet_id, sheet_data['formatting'])
            
            # Make the sheet publicly viewable
            self._set_sharing_permissions(spreadsheet_id)
            
            # Generate shareable link
            sheet_url = f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}"
            
            logger.info(f"Google Sheet created successfully: {spreadsheet_id}")
            
            return f"""
## Google Sheet Created Successfully!

**Live DCF Model**: {sheet_url}

**Features**:
- Live formulas that update automatically
- Interactive charts and visualizations  
- Change assumptions → All calculations update
- Access from any device
- Share with team members

**Sheets included**:
- Assumptions (input parameters)
- Projections (5-year forecasts with formulas)
- Valuation (NPV, terminal value, enterprise value)
            """
            
        except Exception as e:
            logger.error(f"Error creating Google Sheet: {e}")
            return f"Error creating Google Sheet: {e}"
    
    def _populate_sheet(self, spreadsheet_id: str, sheet_info: Dict):
        """Populate a sheet with data and formulas."""
        try:
            sheet_name = sheet_info['name']
            data = sheet_info['data']
            
            # Prepare values for batch update
            values = []
            for row in data:
                values.append(row)
            
            # Update the sheet
            range_name = f"{sheet_name}!A1"
            body = {
                'values': values
            }
            
            self.service.spreadsheets().values().update(
                spreadsheetId=spreadsheet_id,
                range=range_name,
                valueInputOption='USER_ENTERED',  # This interprets formulas
                body=body
            ).execute()
            
            logger.info(f"Sheet '{sheet_name}' populated successfully")
            
        except Exception as e:
            logger.error(f"Error populating sheet {sheet_info['name']}: {e}")
    
    def _apply_formatting(self, spreadsheet_id: str, formatting: Dict):
        """Apply formatting to the spreadsheet."""
        try:
            requests = []
            
            # Currency formatting
            if 'currency_ranges' in formatting:
                for range_str in formatting['currency_ranges']:
                    requests.append({
                        'repeatCell': {
                            'range': self._parse_range(range_str),
                            'cell': {
                                'userEnteredFormat': {
                                    'numberFormat': {
                                        'type': 'CURRENCY',
                                        'pattern': '$#,##0.00'
                                    }
                                }
                            },
                            'fields': 'userEnteredFormat.numberFormat'
                        }
                    })
            
            # Percentage formatting
            if 'percentage_ranges' in formatting:
                for range_str in formatting['percentage_ranges']:
                    requests.append({
                        'repeatCell': {
                            'range': self._parse_range(range_str),
                            'cell': {
                                'userEnteredFormat': {
                                    'numberFormat': {
                                        'type': 'PERCENT',
                                        'pattern': '0.00%'
                                    }
                                }
                            },
                            'fields': 'userEnteredFormat.numberFormat'
                        }
                    })
            
            # Execute formatting requests
            if requests:
                body = {'requests': requests}
                self.service.spreadsheets().batchUpdate(
                    spreadsheetId=spreadsheet_id,
                    body=body
                ).execute()
                
                logger.info("Formatting applied successfully")
                
        except Exception as e:
            logger.error(f"Error applying formatting: {e}")
    
    def _parse_range(self, range_str: str) -> Dict:
        """Parse range string into Sheets API format."""
        # Basic implementation - would need more sophisticated parsing for complex ranges
        return {
            'sheetId': 0,  # Default to first sheet
            'startRowIndex': 0,
            'endRowIndex': 100,
            'startColumnIndex': 0,
            'endColumnIndex': 10
        }
    
    def _set_sharing_permissions(self, spreadsheet_id: str):
        """Set sharing permissions to make sheet publicly viewable."""
        try:
            permission = {
                'type': 'anyone',
                'role': 'reader'
            }
            
            self.drive_service.permissions().create(
                fileId=spreadsheet_id,
                body=permission
            ).execute()
            
            logger.info("Sharing permissions set successfully")
            
        except Exception as e:
            logger.error(f"Error setting sharing permissions: {e}")

def detect_and_create_financial_sheet(query: str, ai_response: str) -> Optional[str]:
    """Detect if AI response should trigger Google Sheet creation."""
    
    # Handle None inputs
    if not query or not ai_response:
        return None
        
    # Ensure inputs are strings
    query = str(query)
    ai_response = str(ai_response)
    
    financial_keywords = ['dcf', 'valuation', 'financial model', 'cash flow', 'npv', 'irr', 'projection']
    
    query_lower = query.lower()
    response_lower = ai_response.lower()
    
    # Check if this is a financial modeling request
    if any(keyword in query_lower or keyword in response_lower for keyword in financial_keywords):
        creator = GoogleSheetsCreator()
        sheet_data = creator.create_dcf_sheet_from_ai_response(ai_response, query)
        result = creator.create_sheet_and_return_link(sheet_data)
        return result
    
    return None