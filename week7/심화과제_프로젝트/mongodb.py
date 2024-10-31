{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "ename": "ModuleNotFoundError",
     "evalue": "No module named 'pymongo'",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mModuleNotFoundError\u001b[0m                       Traceback (most recent call last)",
      "Cell \u001b[1;32mIn[1], line 1\u001b[0m\n\u001b[1;32m----> 1\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01mpymongo\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mmongo_client\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m MongoClient\n\u001b[0;32m      2\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01mpymongo\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mserver_api\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m ServerApi\n\u001b[0;32m      3\u001b[0m \u001b[38;5;66;03m# dotenv 라이브러리 import\u001b[39;00m\n",
      "\u001b[1;31mModuleNotFoundError\u001b[0m: No module named 'pymongo'"
     ]
    }
   ],
   "source": [
    "\n",
    "from pymongo.mongo_client import MongoClient\n",
    "from pymongo.server_api import ServerApi\n",
    "# dotenv 라이브러리 import\n",
    "from dotenv import load_dotenv\n",
    "import os\n",
    "\n",
    "# .env 파일 로드\n",
    "load_dotenv()\n",
    "\n",
    "# 환경 변수 사용\n",
    "mongodb_id = os.getenv(\"mongodb-id\")\n",
    "mongodb_pw = os.getenv(\"mongodb-pw\")\n",
    "\n",
    "\n",
    "uri = f\"mongodb+srv://dosangjib:{mongodb_pw}@devdoe.7ca7a.mongodb.net/?retryWrites=true&w=majority&appName=devdoe\"\n",
    "\n",
    "# Create a new client and connect to the server\n",
    "client = MongoClient(uri, server_api=ServerApi('1'))\n",
    "\n",
    "# Send a ping to confirm a successful connection\n",
    "try:\n",
    "    client.admin.command('ping')\n",
    "    print(\"Pinged your deployment. You successfully connected to MongoDB!\")\n",
    "except Exception as e:\n",
    "    print(e)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": ".venv",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
