"""
Test LOCAGENT GitHub integration functionality.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from src.ue_sea.locagent.github_integration import (
    GitHubConfig, GitHubRepository, github_train, check_github_dependencies
)


class TestGitHubConfig:
    """Test GitHubConfig class."""
    
    def test_config_creation(self):
        """Test GitHub config creation."""
        config = GitHubConfig(
            repo_url="https://github.com/test/repo",
            github_token="test_token",
            issues_limit=50,
            issues_labels=["bug", "enhancement"],
            issues_state="open",
            include_prs=True,
            min_issue_comments=2
        )
        
        assert config.repo_url == "https://github.com/test/repo"
        assert config.github_token == "test_token"
        assert config.issues_limit == 50
        assert config.issues_labels == ["bug", "enhancement"]
        assert config.issues_state == "open"
        assert config.include_prs == True
        assert config.min_issue_comments == 2


class TestGitHubRepository:
    """Test GitHubRepository class."""
    
    def test_parse_repo_url_https(self):
        """Test parsing HTTPS GitHub URL."""
        config = GitHubConfig(repo_url="https://github.com/test/repo")
        repo = GitHubRepository(config)
        
        assert repo.repo_owner == "test"
        assert repo.repo_name == "repo"
    
    def test_parse_repo_url_ssh(self):
        """Test parsing SSH GitHub URL."""
        config = GitHubConfig(repo_url="git@github.com:test/repo.git")
        repo = GitHubRepository(config)
        
        assert repo.repo_owner == "test"
        assert repo.repo_name == "repo"
    
    def test_parse_repo_url_owner_repo(self):
        """Test parsing owner/repo format."""
        config = GitHubConfig(repo_url="test/repo")
        repo = GitHubRepository(config)
        
        assert repo.repo_owner == "test"
        assert repo.repo_name == "repo"
    
    def test_extract_file_references(self):
        """Test file reference extraction."""
        config = GitHubConfig(repo_url="test/repo")
        repo = GitHubRepository(config)
        
        text = """
        The issue is in `src/main.py` and `utils/helper.js`.
        Also check the file test.py and the path src/components/Button.tsx.
        """
        
        file_refs = repo.extract_file_references(text)
        
        assert "src/main.py" in file_refs
        assert "utils/helper.js" in file_refs
        assert "test.py" in file_refs
        assert "src/components/Button.tsx" in file_refs
    
    def test_generate_training_examples(self):
        """Test training example generation."""
        config = GitHubConfig(repo_url="test/repo")
        repo = GitHubRepository(config)
        
        # Mock issues
        issues = [
            {
                "title": "Fix authentication bug",
                "body": "The issue is in `auth/login.py` and `auth/middleware.py`",
                "number": 1,
                "url": "https://github.com/test/repo/issues/1",
                "labels": ["bug"],
                "state": "open",
                "user": "testuser",
                "created_at": "2023-01-01T00:00:00Z",
                "comments_data": []
            },
            {
                "title": "Add new feature",
                "body": "Need to modify `src/features/new.py`",
                "number": 2,
                "url": "https://github.com/test/repo/issues/2",
                "labels": ["enhancement"],
                "state": "open",
                "user": "testuser",
                "created_at": "2023-01-02T00:00:00Z",
                "comments_data": []
            }
        ]
        
        examples = repo.generate_training_examples(issues)
        
        assert len(examples) == 2
        assert examples[0].issue_description == "Fix authentication bug"
        assert "auth/login.py" in examples[0].target_entities
        assert "auth/middleware.py" in examples[0].target_entities
        assert examples[1].issue_description == "Add new feature"
        assert "src/features/new.py" in examples[1].target_entities


class TestGitHubIntegration:
    """Test GitHub integration functionality."""
    
    def test_check_github_dependencies(self):
        """Test GitHub dependency checking."""
        available, missing = check_github_dependencies()
        
        # Should return some result (either available or missing dependencies)
        assert isinstance(available, bool)
        assert isinstance(missing, list)
    
    @pytest.mark.asyncio
    async def test_github_train_mock(self):
        """Test GitHub training with mocked dependencies."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock the GitHub repository and training functions
            with patch('src.ue_sea.locagent.github_integration.GitHubRepository') as mock_repo_class:
                with patch('src.ue_sea.locagent.github_integration.LOCAGENT_Graph') as mock_graph_class:
                    with patch('src.ue_sea.locagent.github_integration.train_locagent_agent') as mock_train:
                        
                        # Setup mocks
                        mock_repo = Mock()
                        mock_repo.clone_repository = AsyncMock(return_value=Path(temp_dir))
                        mock_repo.fetch_issues.return_value = [
                            {
                                "title": "Test issue",
                                "body": "Issue in `test.py`",
                                "number": 1,
                                "url": "https://github.com/test/repo/issues/1",
                                "labels": ["bug"],
                                "state": "open",
                                "user": "testuser",
                                "created_at": "2023-01-01T00:00:00Z",
                                "comments_data": []
                            }
                        ]
                        mock_repo.generate_training_examples.return_value = [
                            Mock(issue_description="Test issue", target_entities=["test.py"])
                        ]
                        mock_repo_class.return_value = mock_repo
                        
                        mock_graph = Mock()
                        mock_graph_class.return_value = mock_graph
                        
                        mock_train.return_value = None
                        
                        # Run GitHub training
                        await github_train(
                            repo_url="https://github.com/test/repo",
                            output_dir=temp_dir,
                            issues_limit=10
                        )
                        
                        # Verify calls
                        mock_repo.clone_repository.assert_called_once()
                        mock_repo.fetch_issues.assert_called_once()
                        mock_repo.generate_training_examples.assert_called_once()
                        mock_graph_class.assert_called_once()
                        mock_train.assert_called_once()


class TestGitHubIntegrationEdgeCases:
    """Test GitHub integration edge cases."""
    
    def test_invalid_repo_url(self):
        """Test handling of invalid repository URLs."""
        config = GitHubConfig(repo_url="invalid-url")
        
        with pytest.raises(ValueError):
            GitHubRepository(config)
    
    def test_empty_issues(self):
        """Test handling of empty issues list."""
        config = GitHubConfig(repo_url="test/repo")
        repo = GitHubRepository(config)
        
        examples = repo.generate_training_examples([])
        assert len(examples) == 0
    
    def test_issues_without_file_references(self):
        """Test handling of issues without file references."""
        config = GitHubConfig(repo_url="test/repo")
        repo = GitHubRepository(config)
        
        issues = [
            {
                "title": "General question",
                "body": "How do I use this feature?",
                "number": 1,
                "url": "https://github.com/test/repo/issues/1",
                "labels": ["question"],
                "state": "open",
                "user": "testuser",
                "created_at": "2023-01-01T00:00:00Z",
                "comments_data": []
            }
        ]
        
        examples = repo.generate_training_examples(issues)
        assert len(examples) == 0  # No file references, so no examples
    
    def test_cleanup(self):
        """Test repository cleanup."""
        config = GitHubConfig(repo_url="test/repo")
        repo = GitHubRepository(config)
        
        # Mock temp repo path
        with tempfile.TemporaryDirectory() as temp_dir:
            repo.temp_repo_path = Path(temp_dir) / "test_repo"
            repo.temp_repo_path.mkdir()
            
            # Verify directory exists
            assert repo.temp_repo_path.exists()
            
            # Cleanup
            repo.cleanup()
            
            # Verify directory is removed
            assert not repo.temp_repo_path.exists()
