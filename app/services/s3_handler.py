import os
import boto3
import io
from PIL import Image
from urllib.parse import urlparse
from botocore.config import Config
from botocore.exceptions import ClientError
from tenacity import retry, stop_after_attempt, wait_exponential
import logging

logger = logging.getLogger(__name__)

class S3Handler:
    """
    Handles S3 storage operations with retry logic.
    Manages file uploads, downloads, and image retrieval.
    """
    
    def __init__(self):
        """
        Initializes S3 client with retry configuration.
        
        Sets up:
            - AWS credentials from environment variables
            - Retry configuration for resilience
            - S3 bucket configuration
            
        Note:
            Requires AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, and AWS_S3_BUCKET
            environment variables to be set
        """
        config = Config(
            retries=dict(
                max_attempts=3,
                mode='adaptive'
            )
        )
        
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
            config=config
        )
        self.bucket_name = os.environ.get('AWS_S3_BUCKET')

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def upload_file_object(self, file_object: bytes, s3_key: str) -> str:
        """
        Uploads a file to S3 with retry logic.
        
        Args:
            file_object (bytes): File data to upload
            s3_key (str): Destination path in S3
            
        Returns:
            str: Public URL of uploaded file
            
        Raises:
            ClientError: If S3 upload fails
            
        Note:
            Implements exponential backoff retry strategy
        """
        try:
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=file_object
            )
            return f"https://{self.bucket_name}.s3.amazonaws.com/{s3_key}"
        except ClientError as e:
            logger.error(f"Error uploading to S3: {str(e)}")
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def download_file_object(self, s3_key: str) -> bytes:
        """
        Downloads a file from S3 with retry logic.
        
        Args:
            s3_key (str): Path of file in S3
            
        Returns:
            bytes: File data
            
        Raises:
            ClientError: If S3 download fails
            
        Note:
            Implements exponential backoff retry strategy
        """
        try:
            response = self.s3_client.get_object(
                Bucket=self.bucket_name,
                Key=s3_key
            )
            return response['Body'].read()
        except ClientError as e:
            logger.error(f"Error downloading from S3: {str(e)}")
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def get_image(self, s3_url: str) -> Image.Image:
        """
        Retrieves and opens an image from S3 with retry logic.
        
        Args:
            s3_url (str): Full S3 URL of the image
            
        Returns:
            Image.Image: PIL Image object in RGB mode
            
        Raises:
            ClientError: If S3 operations fail
            
        Note:
            - Parses S3 URL to extract key
            - Converts image to RGB mode
            - Implements exponential backoff retry strategy
        """
        try:
            parsed_url = urlparse(s3_url)
            key = parsed_url.path.lstrip('/')
            
            response = self.s3_client.get_object(
                Bucket=self.bucket_name,
                Key=key
            )
            image_data = response['Body'].read()
            
            return Image.open(io.BytesIO(image_data)).convert("RGB")
        except ClientError as e:
            logger.error(f"Error downloading image from S3: {str(e)}")
            raise