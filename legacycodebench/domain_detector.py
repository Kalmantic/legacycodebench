"""Domain detection for COBOL programs"""

from typing import Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DomainDetector:
    """Detect business domain from COBOL code analysis"""
    
    def __init__(self):
        self.domain_keywords = {
            "banking": ["account", "balance", "deposit", "withdrawal", "transaction", 
                       "card", "atm", "branch", "customer", "savings", "checking",
                       "overdraft", "statement", "wire", "transfer"],
            "finance": ["interest", "loan", "payment", "credit", "debit", "rate",
                       "principal", "amortization", "portfolio", "investment",
                       "mortgage", "equity", "bond", "security"],
            "insurance": ["policy", "claim", "premium", "coverage", "underwriting",
                         "beneficiary", "insured", "risk", "liability", "actuary",
                         "reinsurance", "adjuster", "deductible"],
            "retail": ["inventory", "order", "customer", "product", "sale", "purchase",
                      "price", "quantity", "warehouse", "shipment", "supplier",
                      "vendor", "stock", "sku"],
            "hr": ["employee", "payroll", "salary", "department", "hire", "benefit",
                  "time", "attendance", "leave", "pension", "performance",
                  "compensation", "workforce"],
            "telecom": ["subscriber", "call", "billing", "network", "service",
                       "usage", "plan", "roaming", "minutes", "data"],
        }
    
    def detect_domain(self, file_analysis: Dict, dataset_name: str = "") -> str:
        """Detect domain from file analysis and dataset context"""
        
        # Check dataset name first (quick)
        dataset_domain = self._detect_from_dataset_name(dataset_name)
        
        # Check code content keywords
        domain_keywords = file_analysis.get("domain_keywords", {})
        
        if not domain_keywords and not dataset_domain:
            return "enterprise"
        
        # If we have keyword counts, use them
        if domain_keywords:
            # Sort by count
            sorted_domains = sorted(domain_keywords.items(), 
                                   key=lambda x: x[1], reverse=True)
            
            if sorted_domains:
                top_domain, top_count = sorted_domains[0]
                
                # If dataset also suggests a domain, prefer dataset if counts are close
                if dataset_domain and dataset_domain in [d for d, _ in sorted_domains[:2]]:
                    return dataset_domain
                
                # Otherwise use top keyword domain if significant
                if top_count >= 3:
                    return top_domain
        
        # Fall back to dataset domain
        if dataset_domain:
            return dataset_domain
        
        # Default
        return "enterprise"
    
    def _detect_from_dataset_name(self, dataset_name: str) -> Optional[str]:
        """Detect domain from dataset/repo name"""
        if not dataset_name:
            return None
        
        name_lower = dataset_name.lower()
        
        # Check for domain keywords in dataset name
        for domain, keywords in self.domain_keywords.items():
            # Check domain name itself
            if domain in name_lower:
                return domain
            
            # Check a few key keywords
            key_keywords = keywords[:3]  # First 3 are usually most distinctive
            if any(keyword in name_lower for keyword in key_keywords):
                return domain
        
        # Special cases
        if "card" in name_lower or "atm" in name_lower:
            return "banking"
        if "demo" in name_lower and "bank" in name_lower:
            return "banking"
        
        return None
    
    def get_domain_description(self, domain: str) -> str:
        """Get human-readable description of domain"""
        descriptions = {
            "banking": "Banking and Financial Services",
            "finance": "Financial Management and Investment",
            "insurance": "Insurance and Risk Management",
            "retail": "Retail and Supply Chain",
            "hr": "Human Resources and Payroll",
            "telecom": "Telecommunications",
            "enterprise": "Enterprise Application",
        }
        return descriptions.get(domain, "General Business")

