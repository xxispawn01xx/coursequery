"""
Excel Generator for Business Valuations and Financial Analysis
Creates downloadable Excel files with DCF models, financial projections, and analysis.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class ExcelGenerator:
    """Generates Excel files for business valuations and financial analysis."""
    
    def __init__(self):
        self.output_dir = Path("temp")
        self.output_dir.mkdir(exist_ok=True)
    
    def create_dcf_model(self, company_data: dict) -> str:
        """Create a DCF (Discounted Cash Flow) valuation model."""
        try:
            # Extract data with defaults
            company_name = company_data.get('name', 'Sample Company')
            revenue_base = company_data.get('revenue', 1000000)
            growth_rate = company_data.get('growth_rate', 0.15)
            margin = company_data.get('margin', 0.20)
            discount_rate = company_data.get('discount_rate', 0.12)
            terminal_growth = company_data.get('terminal_growth', 0.03)
            
            # Create 5-year projection
            years = list(range(2025, 2030))
            projections = []
            
            for i, year in enumerate(years):
                revenue = revenue_base * ((1 + growth_rate) ** i)
                ebitda = revenue * margin
                depreciation = revenue * 0.03  # 3% of revenue
                ebit = ebitda - depreciation
                taxes = ebit * 0.25  # 25% tax rate
                nopat = ebit - taxes
                capex = revenue * 0.05  # 5% of revenue
                working_capital_change = revenue * 0.02  # 2% of revenue
                
                free_cash_flow = nopat + depreciation - capex - working_capital_change
                pv_factor = 1 / ((1 + discount_rate) ** (i + 1))
                present_value = free_cash_flow * pv_factor
                
                projections.append({
                    'Year': year,
                    'Revenue': revenue,
                    'EBITDA': ebitda,
                    'EBIT': ebit,
                    'NOPAT': nopat,
                    'Free Cash Flow': free_cash_flow,
                    'PV Factor': pv_factor,
                    'Present Value': present_value
                })
            
            # Terminal value calculation
            terminal_fcf = projections[-1]['Free Cash Flow'] * (1 + terminal_growth)
            terminal_value = terminal_fcf / (discount_rate - terminal_growth)
            terminal_pv = terminal_value / ((1 + discount_rate) ** 5)
            
            # Enterprise value
            pv_sum = sum(p['Present Value'] for p in projections)
            enterprise_value = pv_sum + terminal_pv
            
            # Create Excel file
            filename = f"DCF_Valuation_{company_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            filepath = self.output_dir / filename
            
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                # Summary sheet
                summary_data = {
                    'Metric': [
                        'Company Name',
                        'Base Revenue (2024)',
                        'Revenue Growth Rate',
                        'EBITDA Margin',
                        'Discount Rate (WACC)',
                        'Terminal Growth Rate',
                        '',
                        'Present Value of FCF (5 years)',
                        'Terminal Value',
                        'Present Value of Terminal Value',
                        'Enterprise Value',
                        'Value per Share (if 1M shares)'
                    ],
                    'Value': [
                        company_name,
                        f"${revenue_base:,.0f}",
                        f"{growth_rate:.1%}",
                        f"{margin:.1%}",
                        f"{discount_rate:.1%}",
                        f"{terminal_growth:.1%}",
                        '',
                        f"${pv_sum:,.0f}",
                        f"${terminal_value:,.0f}",
                        f"${terminal_pv:,.0f}",
                        f"${enterprise_value:,.0f}",
                        f"${enterprise_value/1000000:.2f}"
                    ]
                }
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # Projections sheet
                projections_df = pd.DataFrame(projections)
                projections_df.to_excel(writer, sheet_name='Projections', index=False)
                
                # Sensitivity analysis
                sensitivity_data = []
                discount_rates = [0.08, 0.10, 0.12, 0.14, 0.16]
                growth_rates = [0.10, 0.125, 0.15, 0.175, 0.20]
                
                for dr in discount_rates:
                    row = {'Discount Rate': f"{dr:.1%}"}
                    for gr in growth_rates:
                        # Recalculate with different rates
                        test_projections = []
                        for i in range(5):
                            revenue = revenue_base * ((1 + gr) ** i)
                            fcf = revenue * margin * 0.75  # Simplified FCF
                            pv = fcf / ((1 + dr) ** (i + 1))
                            test_projections.append(pv)
                        
                        terminal_fcf = test_projections[-1] * (1 + terminal_growth)
                        terminal_val = terminal_fcf / (dr - terminal_growth)
                        terminal_pv = terminal_val / ((1 + dr) ** 5)
                        ev = sum(test_projections) + terminal_pv
                        
                        row[f"{gr:.1%}"] = f"${ev:,.0f}"
                    sensitivity_data.append(row)
                
                sensitivity_df = pd.DataFrame(sensitivity_data)
                sensitivity_df.to_excel(writer, sheet_name='Sensitivity', index=False)
            
            logger.info(f"DCF model created: {filename}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error creating DCF model: {e}")
            return None
    
    def create_financial_analysis(self, query_response: str, context_data: dict) -> str:
        """Create financial analysis spreadsheet based on RAG query response."""
        try:
            # Extract key financial metrics from response
            analysis_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Create analysis data
            analysis_data = {
                'Analysis Date': [analysis_date],
                'Query': [context_data.get('query', 'Financial Analysis')],
                'Response Summary': [query_response[:500] + '...' if len(query_response) > 500 else query_response],
                'Source Documents': [str(context_data.get('source_count', 'Multiple'))],
                'Confidence Score': [context_data.get('confidence', 'High')]
            }
            
            # Financial metrics extraction (basic parsing)
            metrics = self._extract_financial_metrics(query_response)
            
            filename = f"Financial_Analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            filepath = self.output_dir / filename
            
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                # Analysis summary
                analysis_df = pd.DataFrame(analysis_data)
                analysis_df.to_excel(writer, sheet_name='Analysis Summary', index=False)
                
                # Financial metrics
                if metrics:
                    metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
                    metrics_df.to_excel(writer, sheet_name='Financial Metrics', index=False)
                
                # Raw response
                response_data = {'Full Response': [query_response]}
                response_df = pd.DataFrame(response_data)
                response_df.to_excel(writer, sheet_name='Full Response', index=False)
            
            logger.info(f"Financial analysis created: {filename}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error creating financial analysis: {e}")
            return None
    
    def _extract_financial_metrics(self, text: str) -> dict:
        """Extract financial metrics from text response."""
        metrics = {}
        
        # Simple regex patterns for common financial terms
        import re
        
        patterns = {
            'Revenue': r'revenue[:\s]+\$?([\d,]+)',
            'Profit': r'profit[:\s]+\$?([\d,]+)',
            'ROI': r'roi[:\s]+([\d.]+)%?',
            'IRR': r'irr[:\s]+([\d.]+)%?',
            'NPV': r'npv[:\s]+\$?([\d,]+)',
            'Cap Rate': r'cap\s+rate[:\s]+([\d.]+)%?',
            'Cash Flow': r'cash\s+flow[:\s]+\$?([\d,]+)',
        }
        
        for metric, pattern in patterns.items():
            match = re.search(pattern, text.lower())
            if match:
                metrics[metric] = match.group(1)
        
        return metrics
    
    def list_generated_files(self) -> list:
        """List all generated Excel files."""
        try:
            excel_files = list(self.output_dir.glob('*.xlsx'))
            return [{'name': f.name, 'path': str(f), 'created': datetime.fromtimestamp(f.stat().st_mtime)} 
                   for f in excel_files]
        except Exception as e:
            logger.error(f"Error listing files: {e}")
            return []