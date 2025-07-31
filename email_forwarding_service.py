#!/usr/bin/env python3
"""
email_forwarding_service.py - Auto-Forwarding Email Service

Creates an ongoing email processing service that can automatically forward
emails to SumMail for processing, enabling real-time email compression
and management.

Features:
- IMAP/SMTP server integration
- Real-time email monitoring
- Intelligent filtering rules
- Auto-forwarding to SumMail
- Background processing service
- Email rule learning system

Author: ototao
License: Apache License 2.0
"""

import os
import time
import json
import logging
import smtpd
import asyncore
import imaplib
import smtplib
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from email import message_from_bytes
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.utils import parseaddr
import schedule
from pathlib import Path

from summail_engine import SumMailEngine, EmailCategory, Priority
from ai_enhanced_interface import EmailFilterEngine, SmartSuggestionEngine

logger = logging.getLogger(__name__)


@dataclass
class ForwardingRule:
    """Email forwarding rule configuration."""
    name: str
    enabled: bool
    conditions: Dict[str, Any]  # sender, subject, content patterns
    actions: Dict[str, Any]     # forward, process, categorize
    priority: int = 0
    learned: bool = False
    accuracy: float = 0.0
    created_date: datetime = None
    last_used: datetime = None


@dataclass
class EmailServiceConfig:
    """Configuration for email forwarding service."""
    # IMAP settings for monitoring
    imap_server: str
    imap_port: int = 993
    imap_use_ssl: bool = True
    
    # SMTP settings for forwarding
    smtp_server: str
    smtp_port: int = 587
    smtp_use_tls: bool = True
    
    # Authentication
    username: str
    password: str
    
    # SumMail processing
    summail_address: str = "summail@localhost"
    auto_process: bool = True
    
    # Monitoring settings
    check_interval: int = 60  # seconds
    max_emails_per_batch: int = 50
    
    # Storage
    rules_file: str = "forwarding_rules.json"
    processed_log: str = "processed_emails.log"


class EmailForwardingService:
    """
    Automated email forwarding service that monitors incoming emails
    and forwards them to SumMail for intelligent processing.
    """
    
    def __init__(self, config: EmailServiceConfig):
        self.config = config
        self.running = False
        self.summail_engine = None
        self.filter_engine = None
        self.rules = []
        self.processed_emails = set()
        self.stats = {
            'emails_monitored': 0,
            'emails_forwarded': 0,
            'emails_processed': 0,
            'rules_applied': 0,
            'start_time': None
        }
        
        self.load_rules()
        self.init_summail()
    
    def load_rules(self):
        """Load forwarding rules from file."""
        rules_path = Path(self.config.rules_file)
        
        if rules_path.exists():
            try:
                with open(rules_path, 'r') as f:
                    rules_data = json.load(f)
                
                self.rules = [
                    ForwardingRule(**rule_data) for rule_data in rules_data
                ]
                
                logger.info(f"Loaded {len(self.rules)} forwarding rules")
                
            except Exception as e:
                logger.error(f"Error loading rules: {e}")
                self.rules = []
        else:
            # Create default rules
            self.create_default_rules()
    
    def save_rules(self):
        """Save forwarding rules to file."""
        try:
            rules_data = []
            for rule in self.rules:
                rule_dict = asdict(rule)
                # Convert datetime objects to ISO strings
                if rule_dict['created_date']:
                    rule_dict['created_date'] = rule_dict['created_date'].isoformat()
                if rule_dict['last_used']:
                    rule_dict['last_used'] = rule_dict['last_used'].isoformat()
                rules_data.append(rule_dict)
            
            with open(self.config.rules_file, 'w') as f:
                json.dump(rules_data, f, indent=2)
                
            logger.info(f"Saved {len(self.rules)} forwarding rules")
            
        except Exception as e:
            logger.error(f"Error saving rules: {e}")
    
    def create_default_rules(self):
        """Create default forwarding rules."""
        default_rules = [
            ForwardingRule(
                name="Newsletter Auto-Forward",
                enabled=True,
                conditions={
                    "subject_contains": ["newsletter", "digest", "update"],
                    "sender_patterns": ["*newsletter*", "*noreply*"]
                },
                actions={
                    "forward_to_summail": True,
                    "category": "newsletter",
                    "priority": "low",
                    "auto_compress": True
                },
                priority=1,
                created_date=datetime.now()
            ),
            
            ForwardingRule(
                name="Software Update Forward",
                enabled=True,
                conditions={
                    "subject_contains": ["update", "version", "release", "patch"],
                    "content_patterns": ["new version", "changelog", "download"]
                },
                actions={
                    "forward_to_summail": True,
                    "category": "software_update",
                    "priority": "medium",
                    "extract_versions": True
                },
                priority=2,
                created_date=datetime.now()
            ),
            
            ForwardingRule(
                name="Financial Email Priority",
                enabled=True,
                conditions={
                    "sender_domains": ["bank", "paypal", "stripe", "payment"],
                    "subject_contains": ["payment", "invoice", "statement", "due"]
                },
                actions={
                    "forward_to_summail": True,
                    "category": "financial",
                    "priority": "high",
                    "immediate_process": True
                },
                priority=5,
                created_date=datetime.now()
            ),
            
            ForwardingRule(
                name="Government Email Critical",
                enabled=True,
                conditions={
                    "sender_domains": [".gov"],
                    "subject_contains": ["tax", "deadline", "notice", "compliance"]
                },
                actions={
                    "forward_to_summail": True,
                    "category": "government",
                    "priority": "critical",
                    "immediate_notify": True
                },
                priority=10,
                created_date=datetime.now()
            )
        ]
        
        self.rules = default_rules
        self.save_rules()
    
    def init_summail(self):
        """Initialize SumMail engine for processing."""
        try:
            self.summail_engine = SumMailEngine({'use_local_ai': True})
            logger.info("SumMail engine initialized")
            
            # Initialize AI filter engine if available
            if hasattr(self.summail_engine, 'ollama_manager'):
                self.filter_engine = EmailFilterEngine(
                    self.summail_engine,
                    self.summail_engine.ollama_manager
                )
                logger.info("AI filter engine initialized")
                
        except Exception as e:
            logger.error(f"Error initializing SumMail: {e}")
    
    def start(self):
        """Start the email forwarding service."""
        if self.running:
            logger.warning("Service is already running")
            return
        
        self.running = True
        self.stats['start_time'] = datetime.now()
        
        logger.info("Starting email forwarding service")
        
        # Start monitoring thread
        monitoring_thread = threading.Thread(target=self._monitor_emails, daemon=True)
        monitoring_thread.start()
        
        # Start scheduled tasks
        self._schedule_tasks()
        
        # Keep service running
        try:
            while self.running:
                schedule.run_pending()
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Service interrupted by user")
            self.stop()
    
    def stop(self):
        """Stop the email forwarding service."""
        self.running = False
        self.save_rules()
        logger.info("Email forwarding service stopped")
    
    def _monitor_emails(self):
        """Monitor emails in background thread."""
        logger.info("Email monitoring started")
        
        while self.running:
            try:
                self._check_new_emails()
                time.sleep(self.config.check_interval)
                
            except Exception as e:
                logger.error(f"Error in email monitoring: {e}")
                time.sleep(self.config.check_interval * 2)  # Back off on error
    
    def _check_new_emails(self):
        """Check for new emails and process them."""
        try:
            # Connect to IMAP server
            if self.config.imap_use_ssl:
                imap = imaplib.IMAP4_SSL(self.config.imap_server, self.config.imap_port)
            else:
                imap = imaplib.IMAP4(self.config.imap_server, self.config.imap_port)
            
            imap.login(self.config.username, self.config.password)
            imap.select('INBOX')
            
            # Search for unprocessed emails
            today = datetime.now().strftime("%d-%b-%Y")
            _, message_ids = imap.search(None, f'SINCE {today}')
            
            if message_ids[0]:
                email_ids = message_ids[0].split()
                new_emails = [eid for eid in email_ids if eid not in self.processed_emails]
                
                # Limit batch size
                new_emails = new_emails[:self.config.max_emails_per_batch]
                
                logger.info(f"Found {len(new_emails)} new emails to process")
                
                for email_id in new_emails:
                    try:
                        self._process_email(imap, email_id)
                        self.processed_emails.add(email_id)
                        self.stats['emails_monitored'] += 1
                        
                    except Exception as e:
                        logger.error(f"Error processing email {email_id}: {e}")
            
            imap.logout()
            
        except Exception as e:
            logger.error(f"Error checking emails: {e}")
    
    def _process_email(self, imap: imaplib.IMAP4, email_id: bytes):
        """Process a single email."""
        try:
            # Fetch email
            _, msg_data = imap.fetch(email_id, '(RFC822)')
            email_message = message_from_bytes(msg_data[0][1])
            
            # Extract email metadata
            sender = parseaddr(email_message.get('From', ''))[1]
            subject = email_message.get('Subject', '')
            
            # Apply forwarding rules
            matched_rules = self._match_rules(email_message, sender, subject)
            
            if matched_rules:
                # Sort by priority (higher priority first)
                matched_rules.sort(key=lambda r: r.priority, reverse=True)
                best_rule = matched_rules[0]
                
                logger.info(f"Email matched rule: {best_rule.name}")
                
                # Execute rule actions
                self._execute_rule_actions(email_message, best_rule)
                
                # Update rule usage
                best_rule.last_used = datetime.now()
                self.stats['rules_applied'] += 1
            
            else:
                # No rules matched - apply default behavior
                logger.info(f"No rules matched for email from {sender}")
                self._handle_unmatched_email(email_message)
        
        except Exception as e:
            logger.error(f"Error processing email: {e}")
    
    def _match_rules(self, email_message, sender: str, subject: str) -> List[ForwardingRule]:
        """Check which rules match the email."""
        matched_rules = []
        
        # Get email body for content matching
        body = self._extract_email_body(email_message)
        sender_domain = sender.split('@')[1] if '@' in sender else ''
        
        for rule in self.rules:
            if not rule.enabled:
                continue
            
            match = True
            conditions = rule.conditions
            
            # Check sender conditions
            if 'sender_patterns' in conditions:
                sender_match = any(
                    pattern.replace('*', '') in sender.lower()
                    for pattern in conditions['sender_patterns']
                )
                if not sender_match:
                    match = False
            
            if 'sender_domains' in conditions:
                domain_match = any(
                    domain in sender_domain.lower()
                    for domain in conditions['sender_domains']
                )
                if not domain_match:
                    match = False
            
            # Check subject conditions
            if 'subject_contains' in conditions:
                subject_match = any(
                    keyword.lower() in subject.lower()
                    for keyword in conditions['subject_contains']
                )
                if not subject_match:
                    match = False
            
            # Check content conditions
            if 'content_patterns' in conditions and body:
                content_match = any(
                    pattern.lower() in body.lower()
                    for pattern in conditions['content_patterns']
                )
                if not content_match:
                    match = False
            
            if match:
                matched_rules.append(rule)
        
        return matched_rules
    
    def _extract_email_body(self, email_message) -> str:
        """Extract text body from email message."""
        body = ""
        
        if email_message.is_multipart():
            for part in email_message.walk():
                if part.get_content_type() == "text/plain":
                    try:
                        body = part.get_payload(decode=True).decode('utf-8', errors='ignore')
                        break
                    except:
                        pass
        else:
            try:
                body = email_message.get_payload(decode=True).decode('utf-8', errors='ignore')
            except:
                pass
        
        return body[:1000]  # First 1000 characters
    
    def _execute_rule_actions(self, email_message, rule: ForwardingRule):
        """Execute actions defined in the forwarding rule."""
        actions = rule.actions
        
        try:
            # Forward to SumMail
            if actions.get('forward_to_summail', False):
                self._forward_to_summail(email_message, rule)
                self.stats['emails_forwarded'] += 1
            
            # Immediate processing
            if actions.get('immediate_process', False):
                self._immediate_process(email_message, rule)
            
            # Immediate notification for critical emails
            if actions.get('immediate_notify', False):
                self._send_notification(email_message, rule)
            
        except Exception as e:
            logger.error(f"Error executing rule actions: {e}")
    
    def _forward_to_summail(self, email_message, rule: ForwardingRule):
        """Forward email to SumMail for processing."""
        try:
            # Create forwarding message
            forward_msg = MIMEMultipart()
            forward_msg['From'] = self.config.username
            forward_msg['To'] = self.config.summail_address
            forward_msg['Subject'] = f"[SumMail-{rule.actions.get('category', 'auto')}] {email_message.get('Subject', '')}"
            
            # Add original email as attachment or inline
            original_body = self._extract_email_body(email_message)
            
            # Add SumMail processing instructions
            instructions = {
                'rule_name': rule.name,
                'category': rule.actions.get('category'),
                'priority': rule.actions.get('priority'),
                'auto_compress': rule.actions.get('auto_compress', False),
                'extract_versions': rule.actions.get('extract_versions', False),
                'original_sender': email_message.get('From'),
                'original_date': email_message.get('Date')
            }
            
            body_text = f"""
SumMail Processing Instructions:
{json.dumps(instructions, indent=2)}

Original Email Body:
{original_body}
            """
            
            forward_msg.attach(MIMEText(body_text, 'plain'))
            
            # Send via SMTP
            self._send_email(forward_msg)
            
            logger.info(f"Email forwarded to SumMail with rule: {rule.name}")
            
        except Exception as e:
            logger.error(f"Error forwarding to SumMail: {e}")
    
    def _immediate_process(self, email_message, rule: ForwardingRule):
        """Immediately process email with SumMail engine."""
        if not self.summail_engine:
            logger.warning("SumMail engine not available for immediate processing")
            return
        
        try:
            # Convert email to bytes for processing
            email_bytes = email_message.as_bytes()
            
            # Process with SumMail
            result = self.summail_engine.process_email(email_bytes)
            
            logger.info(f"Immediate processing complete: {result.metadata.subject}")
            self.stats['emails_processed'] += 1
            
            # Store result or trigger notifications as needed
            self._handle_processed_result(result, rule)
            
        except Exception as e:
            logger.error(f"Error in immediate processing: {e}")
    
    def _send_notification(self, email_message, rule: ForwardingRule):
        """Send notification for critical emails."""
        try:
            notification_msg = MIMEText(f"""
Critical Email Alert - Rule: {rule.name}

From: {email_message.get('From')}
Subject: {email_message.get('Subject')}
Date: {email_message.get('Date')}

This email has been categorized as {rule.actions.get('priority', 'high')} priority.
Please review immediately.
            """)
            
            notification_msg['From'] = self.config.username
            notification_msg['To'] = self.config.username  # Send to self
            notification_msg['Subject'] = f"[CRITICAL] {email_message.get('Subject', '')}"
            
            self._send_email(notification_msg)
            
            logger.info("Critical email notification sent")
            
        except Exception as e:
            logger.error(f"Error sending notification: {e}")
    
    def _handle_unmatched_email(self, email_message):
        """Handle emails that don't match any rules."""
        # Apply default behavior or learn new rules
        
        if self.filter_engine:
            # Use AI to suggest new rules
            try:
                sender = parseaddr(email_message.get('From', ''))[1]
                subject = email_message.get('Subject', '')
                body = self._extract_email_body(email_message)
                
                # This could trigger rule learning
                logger.info(f"Unmatched email from {sender}: {subject[:50]}...")
                
                # TODO: Implement rule learning based on patterns
                
            except Exception as e:
                logger.error(f"Error in AI rule suggestion: {e}")
    
    def _handle_processed_result(self, result, rule: ForwardingRule):
        """Handle the result of email processing."""
        # Update rule accuracy
        if result.compression_ratio > 0.5:  # Good compression
            rule.accuracy = min(rule.accuracy + 0.1, 1.0)
        
        # Log processing result
        with open(self.config.processed_log, 'a') as f:
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'rule_name': rule.name,
                'subject': result.metadata.subject,
                'category': result.metadata.category.value,
                'compression_ratio': result.compression_ratio,
                'processing_time': result.processing_time
            }
            f.write(json.dumps(log_entry) + '\n')
    
    def _send_email(self, message):
        """Send email via SMTP."""
        try:
            if self.config.smtp_use_tls:
                smtp = smtplib.SMTP(self.config.smtp_server, self.config.smtp_port)
                smtp.starttls()
            else:
                smtp = smtplib.SMTP_SSL(self.config.smtp_server, self.config.smtp_port)
            
            smtp.login(self.config.username, self.config.password)
            
            text = message.as_string()
            smtp.sendmail(message['From'], message['To'], text)
            smtp.quit()
            
        except Exception as e:
            logger.error(f"SMTP send error: {e}")
    
    def _schedule_tasks(self):
        """Schedule periodic tasks."""
        # Daily rule optimization
        schedule.every().day.at("02:00").do(self._optimize_rules)
        
        # Weekly statistics report
        schedule.every().sunday.at("09:00").do(self._generate_stats_report)
        
        # Monthly rule cleanup
        schedule.every().month.do(self._cleanup_old_rules)
    
    def _optimize_rules(self):
        """Optimize forwarding rules based on performance."""
        logger.info("Optimizing forwarding rules")
        
        # Disable low-accuracy rules
        for rule in self.rules:
            if rule.learned and rule.accuracy < 0.3:
                rule.enabled = False
                logger.info(f"Disabled low-accuracy rule: {rule.name}")
        
        # TODO: Use AI to suggest rule improvements
        
        self.save_rules()
    
    def _generate_stats_report(self):
        """Generate weekly statistics report."""
        logger.info("Generating statistics report")
        
        uptime = datetime.now() - self.stats['start_time'] if self.stats['start_time'] else timedelta(0)
        
        report = f"""
SumMail Forwarding Service - Weekly Report

Uptime: {uptime}
Emails Monitored: {self.stats['emails_monitored']}
Emails Forwarded: {self.stats['emails_forwarded']}
Emails Processed: {self.stats['emails_processed']}
Rules Applied: {self.stats['rules_applied']}
Active Rules: {len([r for r in self.rules if r.enabled])}
        """
        
        logger.info(report)
        
        # Could send this report via email
    
    def _cleanup_old_rules(self):
        """Clean up old unused rules."""
        logger.info("Cleaning up old rules")
        
        cutoff_date = datetime.now() - timedelta(days=90)
        
        removed_count = 0
        self.rules = [
            rule for rule in self.rules
            if not (rule.learned and 
                   rule.last_used and 
                   rule.last_used < cutoff_date and
                   rule.accuracy < 0.5)
        ]
        
        if removed_count > 0:
            logger.info(f"Removed {removed_count} old unused rules")
            self.save_rules()
    
    def add_rule(self, rule: ForwardingRule):
        """Add a new forwarding rule."""
        rule.created_date = datetime.now()
        self.rules.append(rule)
        self.save_rules()
        logger.info(f"Added new rule: {rule.name}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        uptime = datetime.now() - self.stats['start_time'] if self.stats['start_time'] else timedelta(0)
        
        return {
            **self.stats,
            'uptime_seconds': uptime.total_seconds(),
            'rules_count': len(self.rules),
            'active_rules': len([r for r in self.rules if r.enabled]),
            'learned_rules': len([r for r in self.rules if r.learned]),
            'rule_accuracy': sum(r.accuracy for r in self.rules) / max(len(self.rules), 1)
        }


# Example usage and configuration
def create_gmail_config(username: str, password: str) -> EmailServiceConfig:
    """Create configuration for Gmail account."""
    return EmailServiceConfig(
        imap_server="imap.gmail.com",
        imap_port=993,
        smtp_server="smtp.gmail.com",
        smtp_port=587,
        username=username,
        password=password,
        summail_address="summail@localhost",
        check_interval=300,  # 5 minutes
        max_emails_per_batch=20
    )


def create_outlook_config(username: str, password: str) -> EmailServiceConfig:
    """Create configuration for Outlook/Office365 account."""
    return EmailServiceConfig(
        imap_server="outlook.office365.com",
        imap_port=993,
        smtp_server="smtp.office365.com",
        smtp_port=587,
        username=username,
        password=password,
        summail_address="summail@localhost",
        check_interval=300,
        max_emails_per_batch=20
    )


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python email_forwarding_service.py <email> <password>")
        sys.exit(1)
    
    email_address = sys.argv[1]
    password = sys.argv[2]
    
    # Determine provider and create config
    if "@gmail.com" in email_address:
        config = create_gmail_config(email_address, password)
    elif "@outlook.com" in email_address or "@hotmail.com" in email_address:
        config = create_outlook_config(email_address, password)
    else:
        print("Provider not supported. Use Gmail or Outlook.")
        sys.exit(1)
    
    # Start service
    service = EmailForwardingService(config)
    
    print(f"🚀 Starting SumMail Forwarding Service for {email_address}")
    print("Press Ctrl+C to stop")
    
    try:
        service.start()
    except KeyboardInterrupt:
        print("\n📧 Service stopped by user")
        service.stop()