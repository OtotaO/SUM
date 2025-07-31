#!/usr/bin/env python3
"""
summail_engine.py - Intelligent Email Compression and Analysis Engine

SumMail is an advanced email processing system that intelligently compresses,
categorizes, and summarizes email content to reduce information overload.

Key Features:
- Smart categorization (newsletters, updates, personal, financial, government)
- Duplicate and overlap detection across emails
- Software version extraction and benchmarking
- Newsletter compression with key information extraction
- Privacy-focused local processing
- Temporal analysis for tracking changes over time
- Action item extraction and prioritization

Author: ototao
License: Apache License 2.0
"""

import os
import re
import json
import time
import hashlib
import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from enum import Enum
import email
from email import policy
from email.parser import BytesParser
import imaplib
import smtplib
from pathlib import Path

# Import SUM components
from SUM import HierarchicalDensificationEngine
from multimodal_processor import MultiModalProcessor
from ollama_manager import OllamaManager, ProcessingRequest

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmailCategory(Enum):
    """Email categorization types."""
    NEWSLETTER = "newsletter"
    SOFTWARE_UPDATE = "software_update"
    PROMOTIONAL = "promotional"
    PERSONAL = "personal"
    FINANCIAL = "financial"
    GOVERNMENT = "government"
    WORK = "work"
    SOCIAL_MEDIA = "social_media"
    SUBSCRIPTION = "subscription"
    NOTIFICATION = "notification"
    SPAM = "spam"
    UNKNOWN = "unknown"


class Priority(Enum):
    """Email priority levels."""
    CRITICAL = "critical"      # Government, financial, urgent personal
    HIGH = "high"              # Important work, deadlines
    MEDIUM = "medium"          # Regular updates, newsletters
    LOW = "low"                # Promotional, social media
    ARCHIVE = "archive"        # Old or processed


@dataclass
class EmailMetadata:
    """Metadata extracted from emails."""
    message_id: str
    subject: str
    sender: str
    sender_domain: str
    recipients: List[str]
    date: datetime
    category: EmailCategory
    priority: Priority
    has_attachments: bool
    size: int
    thread_id: Optional[str] = None
    in_reply_to: Optional[str] = None
    references: List[str] = field(default_factory=list)
    extracted_links: List[str] = field(default_factory=list)
    extracted_versions: Dict[str, str] = field(default_factory=dict)
    action_items: List[str] = field(default_factory=list)
    key_dates: List[Tuple[str, datetime]] = field(default_factory=list)
    sentiment_score: float = 0.0
    importance_score: float = 0.0


@dataclass
class SoftwareInfo:
    """Information about software mentioned in emails."""
    name: str
    current_version: str
    latest_version: Optional[str] = None
    release_date: Optional[datetime] = None
    changelog_summary: Optional[str] = None
    benchmark_score: Optional[float] = None
    alternatives: List[str] = field(default_factory=list)
    recommendation: Optional[str] = None


@dataclass
class CompressedEmail:
    """Compressed representation of email content."""
    metadata: EmailMetadata
    summary: str
    key_points: List[str]
    extracted_info: Dict[str, Any]
    similar_emails: List[str]  # Message IDs of similar emails
    compression_ratio: float
    processing_time: float


@dataclass
class EmailDigest:
    """Daily/weekly email digest."""
    period_start: datetime
    period_end: datetime
    total_emails: int
    categories_breakdown: Dict[EmailCategory, int]
    key_updates: List[Dict[str, Any]]
    software_updates: List[SoftwareInfo]
    action_items: List[Dict[str, Any]]
    financial_summary: Optional[Dict[str, Any]] = None
    compressed_newsletters: List[Dict[str, Any]] = field(default_factory=list)
    important_threads: List[Dict[str, Any]] = field(default_factory=list)


class SumMailEngine:
    """
    Advanced email processing engine for intelligent compression and analysis.
    
    This engine processes emails to extract key information, detect patterns,
    and create compressed summaries that reduce information overload.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the SumMail engine."""
        self.config = config or {}
        self.hierarchical_engine = HierarchicalDensificationEngine()
        self.multimodal_processor = MultiModalProcessor()
        self.ollama_manager = OllamaManager() if self.config.get('use_local_ai', True) else None
        
        # Email processing state
        self.processed_emails = {}  # message_id -> CompressedEmail
        self.email_threads = defaultdict(list)  # thread_id -> [message_ids]
        self.software_tracker = {}  # software_name -> SoftwareInfo
        self.newsletter_patterns = {}  # domain -> patterns
        self.duplicate_detector = DuplicateDetector()
        
        # Category patterns
        self._init_category_patterns()
        
        # Software version patterns
        self._init_version_patterns()
        
        logger.info("SumMail Engine initialized")
    
    def _init_category_patterns(self):
        """Initialize email categorization patterns."""
        self.category_patterns = {
            EmailCategory.NEWSLETTER: [
                r'newsletter', r'weekly.*update', r'monthly.*digest',
                r'subscribe', r'unsubscribe', r'email.*preferences'
            ],
            EmailCategory.SOFTWARE_UPDATE: [
                r'new.*version', r'update.*available', r'release.*notes',
                r'changelog', r'v\d+\.\d+', r'patch', r'upgrade'
            ],
            EmailCategory.FINANCIAL: [
                r'payment', r'invoice', r'statement', r'transaction',
                r'balance', r'due.*date', r'billing', r'credit', r'debit'
            ],
            EmailCategory.GOVERNMENT: [
                r'\.gov', r'tax', r'deadline', r'compliance', r'regulatory',
                r'official.*notice', r'government', r'public.*service'
            ],
            EmailCategory.PROMOTIONAL: [
                r'sale', r'discount', r'offer', r'deal', r'save.*%',
                r'limited.*time', r'exclusive', r'promo'
            ],
            EmailCategory.SOCIAL_MEDIA: [
                r'liked.*your', r'commented', r'tagged.*you', r'friend.*request',
                r'notification', r'activity', r'mentioned.*you'
            ]
        }
    
    def _init_version_patterns(self):
        """Initialize software version extraction patterns."""
        self.version_patterns = [
            # Standard version formats
            r'(?:v|version\s*)?(\d+(?:\.\d+){1,3})(?:\s*(?:beta|alpha|rc\d+))?',
            # Date-based versions
            r'(?:version\s*)?(\d{4}[-\.]\d{1,2}[-\.]\d{1,2})',
            # Build numbers
            r'(?:build\s*|b)(\d+(?:\.\d+)*)',
            # Release candidates
            r'(?:RC|rc)(\d+(?:\.\d+)*)'
        ]
        
        # Common software name patterns
        self.software_patterns = [
            r'(?i)([\w\s]+?)\s+(?:version|v)\s*(\d+(?:\.\d+)*)',
            r'(?i)([\w\s]+?)\s+(\d+(?:\.\d+)*)\s+(?:released|available)',
            r'(?i)(?:update|upgrade)\s+(?:to|for)\s+([\w\s]+?)\s+(\d+(?:\.\d+)*)'
        ]
    
    def connect_email_account(self, email_address: str, password: str, 
                            imap_server: str = None, use_oauth: bool = False) -> bool:
        """Connect to email account for processing."""
        try:
            # Auto-detect IMAP server if not provided
            if not imap_server:
                domain = email_address.split('@')[1]
                imap_server = f"imap.{domain}"
                
                # Common provider mappings
                provider_servers = {
                    'gmail.com': 'imap.gmail.com',
                    'outlook.com': 'outlook.office365.com',
                    'yahoo.com': 'imap.mail.yahoo.com',
                    'icloud.com': 'imap.mail.me.com'
                }
                imap_server = provider_servers.get(domain, imap_server)
            
            # Connect to IMAP server
            self.imap = imaplib.IMAP4_SSL(imap_server)
            
            if use_oauth:
                # OAuth2 authentication (requires additional setup)
                # This is a placeholder for OAuth implementation
                raise NotImplementedError("OAuth authentication not yet implemented")
            else:
                self.imap.login(email_address, password)
            
            logger.info(f"Successfully connected to {email_address}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to email account: {e}")
            return False
    
    def process_email(self, email_data: bytes) -> CompressedEmail:
        """Process a single email and create compressed representation."""
        start_time = time.time()
        
        try:
            # Parse email
            msg = BytesParser(policy=policy.default).parsebytes(email_data)
            
            # Extract metadata
            metadata = self._extract_metadata(msg)
            
            # Categorize email
            metadata.category = self._categorize_email(msg, metadata)
            metadata.priority = self._assign_priority(metadata)
            
            # Extract body content
            body_text = self._extract_body(msg)
            
            # Extract structured information
            extracted_info = self._extract_structured_info(body_text, metadata)
            metadata.extracted_links = extracted_info.get('links', [])
            metadata.extracted_versions = extracted_info.get('versions', {})
            metadata.action_items = extracted_info.get('action_items', [])
            metadata.key_dates = extracted_info.get('dates', [])
            
            # Check for duplicates/similar emails
            similar_emails = self.duplicate_detector.find_similar(
                body_text, metadata.subject, metadata.sender_domain
            )
            
            # Generate compressed summary
            if self._should_compress(metadata, body_text):
                summary, key_points = self._generate_compressed_summary(
                    body_text, metadata, extracted_info
                )
            else:
                # For important emails, keep more detail
                summary = body_text[:500] + "..." if len(body_text) > 500 else body_text
                key_points = extracted_info.get('key_points', [])
            
            # Calculate compression ratio
            original_size = len(body_text)
            compressed_size = len(summary) + sum(len(p) for p in key_points)
            compression_ratio = 1 - (compressed_size / max(original_size, 1))
            
            compressed_email = CompressedEmail(
                metadata=metadata,
                summary=summary,
                key_points=key_points,
                extracted_info=extracted_info,
                similar_emails=similar_emails,
                compression_ratio=compression_ratio,
                processing_time=time.time() - start_time
            )
            
            # Store processed email
            self.processed_emails[metadata.message_id] = compressed_email
            
            # Update thread tracking
            if metadata.thread_id:
                self.email_threads[metadata.thread_id].append(metadata.message_id)
            
            # Update software tracking
            if metadata.extracted_versions:
                self._update_software_tracker(metadata.extracted_versions, metadata.date)
            
            return compressed_email
            
        except Exception as e:
            logger.error(f"Error processing email: {e}")
            raise
    
    def _extract_metadata(self, msg: email.message.EmailMessage) -> EmailMetadata:
        """Extract metadata from email message."""
        # Extract sender
        sender = msg.get('From', '')
        sender_match = re.search(r'[\w\.-]+@[\w\.-]+', sender)
        sender_email = sender_match.group(0) if sender_match else sender
        sender_domain = sender_email.split('@')[1] if '@' in sender_email else ''
        
        # Extract recipients
        recipients = []
        for field in ['To', 'Cc']:
            if msg.get(field):
                recipients.extend(re.findall(r'[\w\.-]+@[\w\.-]+', msg.get(field, '')))
        
        # Parse date
        date_str = msg.get('Date', '')
        try:
            from email.utils import parsedate_to_datetime
            date = parsedate_to_datetime(date_str)
        except:
            date = datetime.now()
        
        # Check for attachments
        has_attachments = any(part.get_content_disposition() == 'attachment' 
                            for part in msg.iter_parts())
        
        # Extract thread information
        thread_id = msg.get('Thread-Index') or msg.get('In-Reply-To')
        references = msg.get('References', '').split() if msg.get('References') else []
        
        return EmailMetadata(
            message_id=msg.get('Message-ID', f'unknown_{time.time()}'),
            subject=msg.get('Subject', 'No Subject'),
            sender=sender_email,
            sender_domain=sender_domain,
            recipients=recipients,
            date=date,
            category=EmailCategory.UNKNOWN,
            priority=Priority.MEDIUM,
            has_attachments=has_attachments,
            size=len(str(msg)),
            thread_id=thread_id,
            in_reply_to=msg.get('In-Reply-To'),
            references=references
        )
    
    def _extract_body(self, msg: email.message.EmailMessage) -> str:
        """Extract text body from email."""
        body_parts = []
        
        for part in msg.walk():
            if part.get_content_type() == 'text/plain':
                try:
                    body = part.get_content()
                    if isinstance(body, str):
                        body_parts.append(body)
                except:
                    pass
            elif part.get_content_type() == 'text/html' and not body_parts:
                # Fall back to HTML if no plain text
                try:
                    html_content = part.get_content()
                    if isinstance(html_content, str):
                        # Simple HTML tag removal
                        text = re.sub(r'<[^>]+>', '', html_content)
                        text = re.sub(r'\s+', ' ', text).strip()
                        body_parts.append(text)
                except:
                    pass
        
        return '\n\n'.join(body_parts)
    
    def _categorize_email(self, msg: email.message.EmailMessage, 
                         metadata: EmailMetadata) -> EmailCategory:
        """Categorize email based on content and metadata."""
        subject_lower = metadata.subject.lower()
        sender_lower = metadata.sender.lower()
        body_text = self._extract_body(msg)[:1000].lower()  # First 1000 chars
        
        # Check each category
        category_scores = {}
        
        for category, patterns in self.category_patterns.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, subject_lower):
                    score += 2
                if re.search(pattern, body_text):
                    score += 1
                if re.search(pattern, sender_lower):
                    score += 1
            
            category_scores[category] = score
        
        # Domain-based categorization
        if metadata.sender_domain:
            if '.gov' in metadata.sender_domain:
                category_scores[EmailCategory.GOVERNMENT] += 5
            elif any(bank in metadata.sender_domain for bank in 
                    ['bank', 'paypal', 'stripe', 'square']):
                category_scores[EmailCategory.FINANCIAL] += 5
            elif any(social in metadata.sender_domain for social in 
                    ['facebook', 'twitter', 'linkedin', 'instagram']):
                category_scores[EmailCategory.SOCIAL_MEDIA] += 5
        
        # Get category with highest score
        if category_scores:
            best_category = max(category_scores.items(), key=lambda x: x[1])
            if best_category[1] > 0:
                return best_category[0]
        
        return EmailCategory.UNKNOWN
    
    def _assign_priority(self, metadata: EmailMetadata) -> Priority:
        """Assign priority to email based on category and content."""
        # Critical categories
        if metadata.category in [EmailCategory.GOVERNMENT, EmailCategory.FINANCIAL]:
            return Priority.CRITICAL
        
        # Personal emails are usually high priority
        if metadata.category == EmailCategory.PERSONAL:
            return Priority.HIGH
        
        # Work emails depend on sender and subject
        if metadata.category == EmailCategory.WORK:
            # Check for urgency indicators
            urgent_keywords = ['urgent', 'asap', 'immediately', 'deadline', 'critical']
            if any(keyword in metadata.subject.lower() for keyword in urgent_keywords):
                return Priority.HIGH
            return Priority.MEDIUM
        
        # Newsletters and updates are medium priority
        if metadata.category in [EmailCategory.NEWSLETTER, EmailCategory.SOFTWARE_UPDATE]:
            return Priority.MEDIUM
        
        # Promotional and social media are low priority
        if metadata.category in [EmailCategory.PROMOTIONAL, EmailCategory.SOCIAL_MEDIA]:
            return Priority.LOW
        
        # Spam goes to archive
        if metadata.category == EmailCategory.SPAM:
            return Priority.ARCHIVE
        
        return Priority.MEDIUM
    
    def _extract_structured_info(self, body_text: str, 
                                metadata: EmailMetadata) -> Dict[str, Any]:
        """Extract structured information from email body."""
        extracted = {
            'links': [],
            'versions': {},
            'action_items': [],
            'dates': [],
            'amounts': [],
            'key_points': []
        }
        
        # Extract URLs
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+' 
        extracted['links'] = re.findall(url_pattern, body_text)
        
        # Extract software versions
        for pattern in self.software_patterns:
            matches = re.finditer(pattern, body_text, re.IGNORECASE)
            for match in matches:
                software_name = match.group(1).strip()
                version = match.group(2)
                extracted['versions'][software_name] = version
        
        # Extract action items (look for imperatives and deadlines)
        action_patterns = [
            r'(?:please|kindly|must|should|need to|required to)\s+([^.!?]{10,50})',
            r'(?:deadline|due date|by):\s*([^.!?\n]+)',
            r'(?:action required|response needed|reply by):\s*([^.!?\n]+)'
        ]
        
        for pattern in action_patterns:
            matches = re.finditer(pattern, body_text, re.IGNORECASE)
            for match in matches:
                action = match.group(1).strip()
                if len(action) > 10:
                    extracted['action_items'].append(action)
        
        # Extract dates
        date_patterns = [
            r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            r'(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4})',
            r'((?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),?\s+\d{1,2}\s+\w+)'
        ]
        
        for pattern in date_patterns:
            matches = re.finditer(pattern, body_text, re.IGNORECASE)
            for match in matches:
                date_str = match.group(0)
                # Try to parse the date
                try:
                    # This is simplified - would need better date parsing
                    extracted['dates'].append((date_str, None))
                except:
                    pass
        
        # Extract monetary amounts
        amount_pattern = r'\$?\d+(?:,\d{3})*(?:\.\d{2})?'
        amounts = re.findall(amount_pattern, body_text)
        extracted['amounts'] = [amt for amt in amounts if amt != '']
        
        # Extract key points using hierarchical engine
        if len(body_text) > 100:
            try:
                hierarchical_result = self.hierarchical_engine.process_text(
                    body_text,
                    {'max_concepts': 5, 'max_insights': 3}
                )
                
                if 'key_insights' in hierarchical_result:
                    extracted['key_points'] = [
                        insight['text'] for insight in hierarchical_result['key_insights']
                        if insight.get('score', 0) > 0.5
                    ]
            except:
                pass
        
        return extracted
    
    def _should_compress(self, metadata: EmailMetadata, body_text: str) -> bool:
        """Determine if email should be compressed."""
        # Don't compress critical emails
        if metadata.priority == Priority.CRITICAL:
            return False
        
        # Don't compress very short emails
        if len(body_text) < 200:
            return False
        
        # Compress newsletters and updates
        if metadata.category in [EmailCategory.NEWSLETTER, EmailCategory.SOFTWARE_UPDATE,
                                EmailCategory.PROMOTIONAL, EmailCategory.SOCIAL_MEDIA]:
            return True
        
        # Compress long emails
        if len(body_text) > 1000:
            return True
        
        return False
    
    def _generate_compressed_summary(self, body_text: str, metadata: EmailMetadata,
                                   extracted_info: Dict[str, Any]) -> Tuple[str, List[str]]:
        """Generate compressed summary of email content."""
        try:
            # Use hierarchical engine for initial processing
            hierarchical_result = self.hierarchical_engine.process_text(
                body_text,
                {
                    'max_concepts': 7,
                    'max_summary_tokens': 150,
                    'max_insights': 5
                }
            )
            
            summary = hierarchical_result.get('hierarchical_summary', {}).get('level_2_core', '')
            
            # Enhance with local AI if available
            if self.ollama_manager and self.ollama_manager.available_models:
                try:
                    # Create focused prompt based on category
                    if metadata.category == EmailCategory.NEWSLETTER:
                        prompt = f"Summarize this newsletter, highlighting: 1) Main news/updates 2) Important announcements 3) Relevant dates. Content: {body_text[:1000]}"
                    elif metadata.category == EmailCategory.SOFTWARE_UPDATE:
                        prompt = f"Summarize this software update, highlighting: 1) Version changes 2) New features 3) Bug fixes 4) Breaking changes. Content: {body_text[:1000]}"
                    else:
                        prompt = f"Create a concise summary of this {metadata.category.value} email: {body_text[:1000]}"
                    
                    request = ProcessingRequest(
                        text=prompt,
                        task_type='summarization',
                        max_tokens=200,
                        temperature=0.3
                    )
                    
                    response = self.ollama_manager.process_text(request)
                    if response.response:
                        summary = response.response
                
                except Exception as e:
                    logger.warning(f"Local AI enhancement failed: {e}")
            
            # Extract key points
            key_points = []
            
            # Add extracted action items as key points
            if extracted_info['action_items']:
                key_points.extend([f"Action: {item}" for item in extracted_info['action_items'][:3]])
            
            # Add important dates
            if extracted_info['dates']:
                key_points.extend([f"Date: {date[0]}" for date in extracted_info['dates'][:2]])
            
            # Add software versions
            if extracted_info['versions']:
                for software, version in list(extracted_info['versions'].items())[:2]:
                    key_points.append(f"{software}: v{version}")
            
            # Add monetary amounts if present
            if extracted_info['amounts'] and metadata.category == EmailCategory.FINANCIAL:
                key_points.extend([f"Amount: {amt}" for amt in extracted_info['amounts'][:2]])
            
            return summary, key_points
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            # Fallback to simple truncation
            return body_text[:200] + "...", []
    
    def _update_software_tracker(self, versions: Dict[str, str], 
                               email_date: datetime):
        """Update software version tracking."""
        for software_name, version in versions.items():
            if software_name not in self.software_tracker:
                self.software_tracker[software_name] = SoftwareInfo(
                    name=software_name,
                    current_version=version,
                    release_date=email_date
                )
            else:
                # Update if newer version
                existing = self.software_tracker[software_name]
                if self._compare_versions(version, existing.current_version) > 0:
                    existing.latest_version = version
                    existing.release_date = email_date
    
    def _compare_versions(self, v1: str, v2: str) -> int:
        """Compare version strings. Returns: 1 if v1>v2, -1 if v1<v2, 0 if equal."""
        try:
            # Simple version comparison - would need enhancement for complex versions
            v1_parts = [int(x) for x in v1.split('.')]
            v2_parts = [int(x) for x in v2.split('.')]
            
            for i in range(max(len(v1_parts), len(v2_parts))):
                p1 = v1_parts[i] if i < len(v1_parts) else 0
                p2 = v2_parts[i] if i < len(v2_parts) else 0
                
                if p1 > p2:
                    return 1
                elif p1 < p2:
                    return -1
            
            return 0
        except:
            # Fallback to string comparison
            return 1 if v1 > v2 else (-1 if v1 < v2 else 0)
    
    def generate_digest(self, start_date: datetime, end_date: datetime,
                       categories: List[EmailCategory] = None) -> EmailDigest:
        """Generate email digest for specified period."""
        # Filter emails by date range
        period_emails = [
            email for email in self.processed_emails.values()
            if start_date <= email.metadata.date <= end_date
        ]
        
        # Filter by categories if specified
        if categories:
            period_emails = [
                email for email in period_emails
                if email.metadata.category in categories
            ]
        
        # Category breakdown
        category_breakdown = Counter(
            email.metadata.category for email in period_emails
        )
        
        # Extract key updates
        key_updates = []
        
        # Software updates
        software_updates = []
        for email in period_emails:
            if email.metadata.category == EmailCategory.SOFTWARE_UPDATE:
                for software, version in email.metadata.extracted_versions.items():
                    if software in self.software_tracker:
                        software_updates.append(self.software_tracker[software])
        
        # Collect action items
        all_action_items = []
        for email in period_emails:
            for action in email.metadata.action_items:
                all_action_items.append({
                    'action': action,
                    'source': email.metadata.subject,
                    'date': email.metadata.date,
                    'priority': email.metadata.priority.value
                })
        
        # Sort action items by priority and date
        all_action_items.sort(
            key=lambda x: (
                ['critical', 'high', 'medium', 'low'].index(x['priority']),
                x['date']
            )
        )
        
        # Compress newsletters
        newsletter_emails = [
            email for email in period_emails
            if email.metadata.category == EmailCategory.NEWSLETTER
        ]
        
        compressed_newsletters = self._compress_newsletters(newsletter_emails)
        
        # Identify important threads
        important_threads = self._identify_important_threads(period_emails)
        
        # Financial summary if applicable
        financial_emails = [
            email for email in period_emails
            if email.metadata.category == EmailCategory.FINANCIAL
        ]
        
        financial_summary = self._generate_financial_summary(financial_emails) if financial_emails else None
        
        return EmailDigest(
            period_start=start_date,
            period_end=end_date,
            total_emails=len(period_emails),
            categories_breakdown=dict(category_breakdown),
            key_updates=key_updates,
            software_updates=software_updates,
            action_items=all_action_items[:10],  # Top 10 action items
            financial_summary=financial_summary,
            compressed_newsletters=compressed_newsletters,
            important_threads=important_threads
        )
    
    def _compress_newsletters(self, newsletters: List[CompressedEmail]) -> List[Dict[str, Any]]:
        """Compress multiple newsletters into consolidated summaries."""
        # Group by sender domain
        by_sender = defaultdict(list)
        for email in newsletters:
            by_sender[email.metadata.sender_domain].append(email)
        
        compressed = []
        
        for domain, emails in by_sender.items():
            # Combine summaries
            all_points = []
            for email in emails:
                all_points.extend(email.key_points)
            
            # Remove duplicates while preserving order
            seen = set()
            unique_points = []
            for point in all_points:
                if point not in seen:
                    seen.add(point)
                    unique_points.append(point)
            
            compressed.append({
                'source': domain,
                'email_count': len(emails),
                'date_range': f"{min(e.metadata.date for e in emails).strftime('%Y-%m-%d')} to "
                             f"{max(e.metadata.date for e in emails).strftime('%Y-%m-%d')}",
                'key_points': unique_points[:10],  # Top 10 points
                'subjects': [e.metadata.subject for e in emails[:3]]  # Sample subjects
            })
        
        return compressed
    
    def _identify_important_threads(self, emails: List[CompressedEmail]) -> List[Dict[str, Any]]:
        """Identify important email threads."""
        thread_importance = defaultdict(float)
        thread_emails = defaultdict(list)
        
        for email in emails:
            if email.metadata.thread_id:
                # Calculate thread importance
                importance = 0
                
                # Priority-based importance
                priority_scores = {
                    Priority.CRITICAL: 5,
                    Priority.HIGH: 3,
                    Priority.MEDIUM: 1,
                    Priority.LOW: 0.5,
                    Priority.ARCHIVE: 0
                }
                importance += priority_scores.get(email.metadata.priority, 1)
                
                # Category-based importance
                category_scores = {
                    EmailCategory.PERSONAL: 3,
                    EmailCategory.WORK: 2,
                    EmailCategory.FINANCIAL: 3,
                    EmailCategory.GOVERNMENT: 4
                }
                importance += category_scores.get(email.metadata.category, 1)
                
                # Action items add importance
                importance += len(email.metadata.action_items) * 2
                
                thread_importance[email.metadata.thread_id] += importance
                thread_emails[email.metadata.thread_id].append(email)
        
        # Sort threads by importance
        important_threads = sorted(
            thread_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]  # Top 5 threads
        
        thread_summaries = []
        for thread_id, importance in important_threads:
            emails_in_thread = thread_emails[thread_id]
            thread_summaries.append({
                'thread_id': thread_id,
                'importance_score': importance,
                'email_count': len(emails_in_thread),
                'participants': list(set(e.metadata.sender for e in emails_in_thread)),
                'date_range': f"{min(e.metadata.date for e in emails_in_thread).strftime('%Y-%m-%d')} to "
                             f"{max(e.metadata.date for e in emails_in_thread).strftime('%Y-%m-%d')}",
                'subject': emails_in_thread[0].metadata.subject,
                'key_points': [
                    point for email in emails_in_thread
                    for point in email.key_points[:2]
                ][:10]
            })
        
        return thread_summaries
    
    def _generate_financial_summary(self, financial_emails: List[CompressedEmail]) -> Dict[str, Any]:
        """Generate summary of financial emails."""
        summary = {
            'total_emails': len(financial_emails),
            'transactions': [],
            'total_amounts': {},
            'due_dates': [],
            'action_required': []
        }
        
        for email in financial_emails:
            # Extract transaction amounts
            if email.extracted_info.get('amounts'):
                for amount in email.extracted_info['amounts']:
                    summary['transactions'].append({
                        'amount': amount,
                        'source': email.metadata.sender_domain,
                        'date': email.metadata.date.strftime('%Y-%m-%d'),
                        'subject': email.metadata.subject
                    })
            
            # Extract due dates
            if email.extracted_info.get('dates'):
                for date_str, parsed_date in email.extracted_info['dates']:
                    if 'due' in email.summary.lower() or 'payment' in email.summary.lower():
                        summary['due_dates'].append({
                            'date': date_str,
                            'source': email.metadata.sender_domain,
                            'subject': email.metadata.subject
                        })
            
            # Extract action items
            if email.metadata.action_items:
                summary['action_required'].extend([
                    {
                        'action': action,
                        'source': email.metadata.sender_domain,
                        'priority': email.metadata.priority.value
                    }
                    for action in email.metadata.action_items
                ])
        
        return summary


class DuplicateDetector:
    """Detect duplicate and similar emails."""
    
    def __init__(self, similarity_threshold: float = 0.8):
        """Initialize duplicate detector."""
        self.similarity_threshold = similarity_threshold
        self.content_hashes = {}  # hash -> [message_ids]
        self.subject_index = defaultdict(list)  # subject_words -> [message_ids]
        
    def find_similar(self, content: str, subject: str, sender_domain: str) -> List[str]:
        """Find similar emails based on content and metadata."""
        similar_emails = []
        
        # Generate content hash
        content_hash = self._generate_hash(content)
        
        # Check exact duplicates
        if content_hash in self.content_hashes:
            similar_emails.extend(self.content_hashes[content_hash])
        
        # Check similar subjects
        subject_words = set(subject.lower().split())
        for word in subject_words:
            if len(word) > 3:  # Skip short words
                similar_emails.extend(self.subject_index[word])
        
        # Remove duplicates and return
        return list(set(similar_emails))
    
    def add_email(self, message_id: str, content: str, subject: str):
        """Add email to duplicate detection index."""
        # Add content hash
        content_hash = self._generate_hash(content)
        if content_hash not in self.content_hashes:
            self.content_hashes[content_hash] = []
        self.content_hashes[content_hash].append(message_id)
        
        # Add to subject index
        subject_words = set(subject.lower().split())
        for word in subject_words:
            if len(word) > 3:
                self.subject_index[word].append(message_id)
    
    def _generate_hash(self, content: str) -> str:
        """Generate hash of content for duplicate detection."""
        # Normalize content
        normalized = re.sub(r'\s+', ' ', content.lower().strip())
        # Remove common variations (dates, greetings, etc.)
        normalized = re.sub(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', 'DATE', normalized)
        normalized = re.sub(r'dear\s+\w+', 'GREETING', normalized)
        
        return hashlib.md5(normalized.encode()).hexdigest()


# Example usage and testing
if __name__ == "__main__":
    # Initialize SumMail engine
    engine = SumMailEngine({
        'use_local_ai': True,
        'compression_level': 'high'
    })
    
    print("SumMail Engine initialized!")
    print("\nCapabilities:")
    print("- Smart email categorization")
    print("- Duplicate detection")
    print("- Software version tracking") 
    print("- Newsletter compression")
    print("- Action item extraction")
    print("- Financial summary generation")
    print("- Thread importance analysis")
    
    # Example email processing
    sample_email = b"""From: updates@software.com
To: user@example.com
Subject: New Version 2.5.0 Released - Important Security Update
Date: Mon, 15 Jan 2024 10:00:00 -0000

Dear User,

We're excited to announce the release of Software Suite v2.5.0!

This update includes:
- Critical security patches for CVE-2024-1234
- New dashboard interface with improved performance
- Bug fixes for the export functionality
- Enhanced API rate limiting

Please update by January 31st to ensure continued security.

Download: https://software.com/download/v2.5.0

Best regards,
The Software Team
"""
    
    try:
        result = engine.process_email(sample_email)
        print(f"\nProcessed email:")
        print(f"Category: {result.metadata.category.value}")
        print(f"Priority: {result.metadata.priority.value}")
        print(f"Summary: {result.summary}")
        print(f"Key points: {result.key_points}")
        print(f"Compression ratio: {result.compression_ratio:.2%}")
    except Exception as e:
        print(f"Processing failed: {e}")