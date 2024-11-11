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

# Copy the entire S3Handler class from the original code
class S3Handler:
    def __init__(self):
        """Initialize S3 client with retry configuration."""
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
        """Upload a file object to S3 with retry logic."""
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
        """Download a file object from S3 with retry logic."""
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
        """Get image from S3 URL with retry logic."""
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
            raise        # ... (copy the entire S3Handler class implementation)