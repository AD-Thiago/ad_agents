# core/domains/__init__.py

from .market_segments import MarketSegmentManager, MarketSegment, MarketPriority, Industry
from .seasonality import SeasonalityManager, SeasonalEvent, SeasonType
from .tech_stacks import TechStackManager, TechStack, Framework, TechCategory, MaturityLevel
from .bigtech_monitor import BigTechMonitor, BigTechCompany, Innovation, CompanyCategory
from .adoption_metrics import AdoptionMetricsManager, AdoptionMetric, ROIMetric, UseCaseMetric
from .definitions import DomainManager, DomainDefinition, DomainType, ContentType, ValidationRule

__all__ = [
   # Market Segments
   'MarketSegmentManager',
   'MarketSegment',
   'MarketPriority',
   'Industry',
   
   # Seasonality
   'SeasonalityManager',
   'SeasonalEvent',
   'SeasonType',
   
   # Tech Stacks
   'TechStackManager',
   'TechStack',
   'Framework', 
   'TechCategory',
   'MaturityLevel',
   
   # BigTech Monitor
   'BigTechMonitor',
   'BigTechCompany',
   'Innovation',
   'CompanyCategory',
   
   # Adoption Metrics
   'AdoptionMetricsManager',
   'AdoptionMetric',
   'ROIMetric',
   'UseCaseMetric',
   
   # Definitions
   'DomainManager',
   'DomainDefinition',
   'DomainType',
   'ContentType',
   'ValidationRule'
]