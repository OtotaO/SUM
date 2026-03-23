#!/usr/bin/env python3
"""
API Key Management CLI for SUM

Command-line tool for managing API keys.

Usage:
    python manage_api_keys.py list
    python manage_api_keys.py create <name> [--permissions=read,summarize] [--rate-limit=60]
    python manage_api_keys.py revoke <key_id>
    python manage_api_keys.py stats <key_id> [--days=7]

Author: SUM Team
License: Apache License 2.0
"""

import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from api.auth import get_auth_manager


def list_keys(args):
    """List all API keys."""
    manager = get_auth_manager()
    keys = manager.list_keys()
    
    if not keys:
        print("No API keys found.")
        return
        
    print(f"\n{'ID':<20} {'Name':<30} {'Active':<8} {'Rate':<10} {'Daily':<10} {'Total Reqs':<12}")
    print("-" * 100)
    
    for key in keys:
        print(f"{key['key_id']:<20} {key['name']:<30} "
              f"{'Yes' if key['is_active'] else 'No':<8} "
              f"{key['rate_limit']:<10} {key['daily_limit']:<10} "
              f"{key['total_requests']:<12}")
    
    print(f"\nTotal keys: {len(keys)}")


def create_key(args):
    """Create a new API key."""
    manager = get_auth_manager()
    
    permissions = args.permissions.split(',') if args.permissions else ['read', 'summarize']
    
    try:
        key_id, api_key = manager.generate_api_key(
            name=args.name,
            permissions=permissions,
            rate_limit=args.rate_limit,
            daily_limit=args.daily_limit
        )
        
        print("\n✓ API Key created successfully!")
        print(f"\nKey ID: {key_id}")
        print(f"API Key: {api_key}")
        print(f"Name: {args.name}")
        print(f"Permissions: {', '.join(permissions)}")
        print(f"Rate Limit: {args.rate_limit} requests/minute")
        print(f"Daily Limit: {args.daily_limit} requests/day")
        print("\n⚠️  Save this API key securely - it cannot be retrieved again!")
        
        # Optionally save to file
        if args.save_to_file:
            filename = f"api_key_{key_id}.txt"
            with open(filename, 'w') as f:
                f.write(f"API Key Details\n")
                f.write(f"===============\n")
                f.write(f"Created: {datetime.now().isoformat()}\n")
                f.write(f"Key ID: {key_id}\n")
                f.write(f"API Key: {api_key}\n")
                f.write(f"Name: {args.name}\n")
                f.write(f"Permissions: {', '.join(permissions)}\n")
                f.write(f"Rate Limit: {args.rate_limit} requests/minute\n")
                f.write(f"Daily Limit: {args.daily_limit} requests/day\n")
            
            Path(filename).chmod(0o600)  # Restrict file permissions
            print(f"\nKey saved to: {filename}")
            
    except Exception as e:
        print(f"\n✗ Error creating key: {e}")
        sys.exit(1)


def revoke_key(args):
    """Revoke an API key."""
    manager = get_auth_manager()
    
    try:
        manager.revoke_key(args.key_id)
        print(f"\n✓ API key {args.key_id} revoked successfully!")
    except Exception as e:
        print(f"\n✗ Error revoking key: {e}")
        sys.exit(1)


def show_stats(args):
    """Show usage statistics for an API key."""
    manager = get_auth_manager()
    
    try:
        stats = manager.get_usage_stats(args.key_id, args.days)
        
        print(f"\nUsage Statistics for Key: {args.key_id}")
        print(f"Period: Last {args.days} days")
        print("=" * 50)
        
        print(f"\nTotal Requests: {stats['total_requests']:,}")
        print(f"Average Response Time: {stats['avg_response_time']:.3f}s")
        print(f"Error Rate: {stats['error_rate']:.2f}%")
        
        if stats['endpoints']:
            print("\nTop Endpoints:")
            for endpoint, count in list(stats['endpoints'].items())[:10]:
                print(f"  {endpoint}: {count:,} requests")
                
    except Exception as e:
        print(f"\n✗ Error getting stats: {e}")
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='API Key Management for SUM',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python manage_api_keys.py list
  python manage_api_keys.py create "My App" --permissions=read,summarize --rate-limit=100
  python manage_api_keys.py revoke abc123
  python manage_api_keys.py stats abc123 --days=30
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List all API keys')
    
    # Create command
    create_parser = subparsers.add_parser('create', help='Create a new API key')
    create_parser.add_argument('name', help='Friendly name for the key')
    create_parser.add_argument('--permissions', default='read,summarize',
                              help='Comma-separated permissions (default: read,summarize)')
    create_parser.add_argument('--rate-limit', type=int, default=60,
                              help='Requests per minute (default: 60)')
    create_parser.add_argument('--daily-limit', type=int, default=10000,
                              help='Requests per day (default: 10000)')
    create_parser.add_argument('--save-to-file', action='store_true',
                              help='Save key details to a file')
    
    # Revoke command
    revoke_parser = subparsers.add_parser('revoke', help='Revoke an API key')
    revoke_parser.add_argument('key_id', help='Key ID to revoke')
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show usage statistics')
    stats_parser.add_argument('key_id', help='Key ID to show stats for')
    stats_parser.add_argument('--days', type=int, default=7,
                             help='Number of days to show (default: 7)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
        
    # Execute command
    if args.command == 'list':
        list_keys(args)
    elif args.command == 'create':
        create_key(args)
    elif args.command == 'revoke':
        revoke_key(args)
    elif args.command == 'stats':
        show_stats(args)


if __name__ == '__main__':
    main()