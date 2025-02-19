import os
import json
import argparse
from typing import Dict, Optional, List
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path
import requests

# Load environment variables from .env file
load_dotenv()

class GitHubAnalyzer:
    def __init__(self):
        """Initialize the GitHub analyzer."""
        self.token = os.getenv('GITHUB_TOKEN')
        if not self.token:
            raise ValueError("GitHub token is required. Set it as GITHUB_TOKEN environment variable.")
        
        self.headers = {
            'Authorization': f'Bearer {self.token}',
            'Content-Type': 'application/json',
        }
        self.api_url = 'https://api.github.com/graphql'

    def get_closed_issues_with_prs(self, repo_owner: str, repo_name: str, max_issues: Optional[int] = None) -> List[Dict]:
        """
        Get all closed issues that have associated merged pull requests using GraphQL.
        
        Args:
            repo_owner (str): The owner of the repository
            repo_name (str): The name of the repository
            max_issues (Optional[int]): Maximum number of issues to process
            
        Returns:
            List[Dict]: List of issues with their associated merged PRs
        """
        query = """
        query($owner: String!, $name: String!, $cursor: String) {
          repository(owner: $owner, name: $name) {
            issues(first: 100, after: $cursor, states: [CLOSED], orderBy: {field: CREATED_AT, direction: DESC}) {
              pageInfo {
                hasNextPage
                endCursor
              }
              nodes {
                number
                title
                url
                closedAt
                author {
                  login
                }
                body
                comments(first: 100) {
                  nodes {
                    author {
                      login
                    }
                    body
                    createdAt
                  }
                }
                timelineItems(first: 100, itemTypes: [CROSS_REFERENCED_EVENT]) {
                  nodes {
                    ... on CrossReferencedEvent {
                      source {
                        ... on PullRequest {
                          number
                          title
                          url
                          merged
                          mergedAt
                          baseRefOid
                          baseRefName
                          baseRepository {
                            nameWithOwner
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
        """
        
        issues_with_prs = []
        cursor = None
        total_issues_processed = 0
        
        while True:
            # Check if we've reached the maximum
            if max_issues and total_issues_processed >= max_issues:
                print(f"Reached maximum number of issues to process ({max_issues})")
                break

            # Prepare variables for the query
            variables = {
                "owner": repo_owner,
                "name": repo_name,
                "cursor": cursor
            }
            
            # Make the GraphQL request
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json={"query": query, "variables": variables}
            )
            
            if response.status_code != 200:
                raise Exception(f"GraphQL query failed: {response.status_code} - {response.text}")
            
            data = response.json()
            if "errors" in data:
                raise Exception(f"GraphQL query returned errors: {data['errors']}")
            
            # Process the issues
            repository = data["data"]["repository"]
            if not repository:
                raise Exception(f"Repository {repo_owner}/{repo_name} not found")
                
            issues_data = repository["issues"]
            
            for issue in issues_data["nodes"]:
                # Check if we've reached the maximum
                if max_issues and total_issues_processed >= max_issues:
                    break

                total_issues_processed += 1
                
                # Find merged PRs in the timeline
                merged_prs = []
                seen_prs = set()
                
                for timeline_item in issue["timelineItems"]["nodes"]:
                    source = timeline_item.get("source", {})
                    if (source and 
                        source.get("merged") and 
                        source.get("number") not in seen_prs):
                        
                        # Only include PRs from the same repository
                        if source.get("baseRepository", {}).get("nameWithOwner") == f"{repo_owner}/{repo_name}":
                            merged_prs.append({
                                "pr_number": source["number"],
                                "pr_title": source["title"],
                                "pr_url": source["url"],
                                "merged_at": source["mergedAt"],
                                "base_commit": {
                                    "sha": source["baseRefOid"],
                                    "ref": source["baseRefName"],
                                }
                            })
                            seen_prs.add(source["number"])
                
                if merged_prs:
                    # Process comments
                    discussion = [{
                        "author": issue.get("author", {}).get("login"),
                        "body": issue["body"],
                        "type": "issue"
                    }]
                    
                    for comment in issue.get("comments", {}).get("nodes", []):
                        if comment.get("author"):  # Skip comments from deleted users
                            discussion.append({
                                "author": comment["author"]["login"],
                                "body": comment["body"],
                                "created_at": comment["createdAt"],
                                "type": "comment"
                            })
                    
                    issues_with_prs.append({
                        "repository": f"{repo_owner}/{repo_name}",
                        "issue_number": issue["number"],
                        "issue_title": issue["title"],
                        "issue_url": issue["url"],
                        "closed_at": issue["closedAt"],
                        "merged_prs": merged_prs,
                        "discussion": discussion
                    })
            
            # Check if we should continue to next page
            if max_issues and total_issues_processed >= max_issues:
                break
                
            page_info = issues_data["pageInfo"]
            if not page_info["hasNextPage"]:
                break
                
            cursor = page_info["endCursor"]
            print(f"Processed {total_issues_processed} issues, found {len(issues_with_prs)} with PRs...")
        
        return issues_with_prs

def load_projects(projects_file: Path) -> List[Dict[str, str]]:
    """Load projects from a JSON file."""
    try:
        with open(projects_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if not isinstance(data, dict) or 'projects' not in data:
                raise ValueError("Invalid projects.json format. Must contain a 'projects' array.")
            return [
                {'owner': owner, 'name': name}
                for project in data['projects']
                for owner, name in [project.split('/')]
            ]
    except FileNotFoundError:
        raise FileNotFoundError(f"Projects file not found: {projects_file}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON in projects file: {projects_file}")

def parse_positive_int(value: str) -> int:
    """Parse a positive integer."""
    try:
        ivalue = int(value)
        if ivalue <= 0:
            raise ValueError
        return ivalue
    except ValueError:
        raise argparse.ArgumentTypeError(
            "Must be a positive integer"
        )

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='Find closed issues with merged pull requests in GitHub repositories.'
    )
    parser.add_argument(
        '-p', '--projects-file',
        type=Path,
        default='projects.json',
        help='JSON file containing projects to analyze (default: projects.json)'
    )
    parser.add_argument(
        '-o', '--output',
        type=Path,
        default='issues.json',
        help='Output JSON file (default: issues.json)'
    )
    parser.add_argument(
        '-m', '--max-issues',
        default=500,
        type=parse_positive_int,
        help='Maximum number of issues to process per project'
    )

    args = parser.parse_args()
    
    try:
        # Load projects
        projects = load_projects(args.projects_file)
        print(f"Loaded {len(projects)} projects from {args.projects_file}")
        
        # Initialize the analyzer
        analyzer = GitHubAnalyzer()
        
        # Process all projects and collect issues
        all_issues = []
        
        for project in projects:
            repo_owner = project['owner']
            repo_name = project['name']
            
            print(f"\nAnalyzing {repo_owner}/{repo_name}...")
            print(f"Will process up to {args.max_issues} issues")
            
            try:
                # Get issues with merged PRs
                issues = analyzer.get_closed_issues_with_prs(repo_owner, repo_name, args.max_issues)
                all_issues.extend(issues)
                
                print(f"Found {len(issues)} issues with merged PRs")
                
            except Exception as e:
                print(f"Error processing {repo_owner}/{repo_name}: {str(e)}")
                print("Continuing with next project...")
                continue
        
        # Prepare the output data
        output_data = {
            'total_issues': len(all_issues),
            'projects_analyzed': len(projects),
            'generated_at': datetime.utcnow().isoformat(),
            'issues': all_issues
        }
        
        # Create output directory if needed
        args.output.parent.mkdir(parents=True, exist_ok=True)
        
        # Write all issues to a single file
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
            
        print(f"\nTotal issues found across all projects: {len(all_issues)}")
        print(f"Results written to {args.output}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        parser.exit(1)

if __name__ == "__main__":
    main() 