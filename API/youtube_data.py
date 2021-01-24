__author__ = "Y-Nots"

from collections import defaultdict
import json
import pandas as pd
from utils.helper import openURL


class VideoData:
    def __init__(self, videoId, key):

        self.metadatas = defaultdict(list)
        self.comments = defaultdict(list)
        self.replies = defaultdict(list)
        self.key = key
        self.metaDataParams = {

            'part': 'snippet,statistics',
            'maxResults': 10,
            'id': videoId,
            'textFormat': 'plainText',
            'key': key
        }

        self.commentParams = {
            'part': 'snippet,replies',
            'maxResults': 100,
            'videoId': videoId,
            'textFormat': 'plainText',
            'key': key
        }

    def load_comments(self, mat):
        for item in mat["items"]:
            comment = item["snippet"]["topLevelComment"]
            self.comments["id"].append(comment["id"])
            self.comments["comment"].append(comment["snippet"]["textDisplay"])
            self.comments["author"].append(comment["snippet"]["authorDisplayName"])
            self.comments["likeCount"].append(comment["snippet"]["likeCount"])
            self.comments["totalReplyCount"].append(item["snippet"]["totalReplyCount"])
            self.comments["publishedAt"].append(comment["snippet"]["publishedAt"])
            if int(item["snippet"]["totalReplyCount"]) > 5:
                parentid = comment["id"]
                self.get_video_replies(parentid)
            else:
                if 'replies' in item.keys():
                    for reply in item['replies']['comments']:
                        self.replies["parentId"].append(reply["snippet"]["parentId"])
                        self.replies["authorDisplayName"].append(reply['snippet']['authorDisplayName'])
                        self.replies["replyComment"].append(reply["snippet"]["textDisplay"])
                        self.replies["publishedAt"].append(reply["snippet"]["publishedAt"])
                        self.replies["likeCount"].append(reply["snippet"]["likeCount"])
            df = pd.DataFrame().from_dict(self.replies)
            self.comments["replies"].append(df)
            self.replies = defaultdict(list)

    def load_metadata(self, mat):
        for item in mat["items"]:
            metadata = item
            self.metadatas["channelId"].append(metadata["snippet"]["channelId"])
            self.metadatas["publishedAt"].append(metadata["snippet"]["publishedAt"])
            self.metadatas["title"].append(metadata["snippet"]["title"])
            self.metadatas["description"].append(metadata["snippet"]["description"])
            self.metadatas["viewCount"].append(metadata["statistics"]["viewCount"])
            self.metadatas["likeCount"].append(metadata["statistics"]["likeCount"])
            self.metadatas["dislikeCount"].append(metadata["statistics"]["dislikeCount"])
            self.metadatas["commentCount"].append(metadata["statistics"]["commentCount"])
            if 'tags' in item["snippet"].keys():
                self.metadatas["tags"].append(metadata["snippet"]["tags"])
            self.metadatas["thumbnailUrl"].append(metadata["snippet"]["thumbnails"]["high"]["url"])
            df1 = pd.DataFrame().from_dict(self.comments)
            self.metadatas["comments"].append(df1)




    def get_video_comments(self):
        url_response = json.loads(openURL("https://www.googleapis.com/youtube/v3/commentThreads", self.commentParams))
        nextPageToken = url_response.get("nextPageToken")
        self.load_comments(url_response)

        while nextPageToken:
            self.commentParams.update({'pageToken': nextPageToken})
            url_response = json.loads(
                openURL("https://www.googleapis.com/youtube/v3/commentThreads", self.commentParams))
            nextPageToken = url_response.get("nextPageToken")
            self.load_comments(url_response)
        self.get_video_metadata()

    def get_video_metadata(self):
        url_response = json.loads(openURL("https://www.googleapis.com/youtube/v3/videos", self.metaDataParams))
        nextPageToken = url_response.get("nextPageToken")
        self.load_metadata(url_response)

        while nextPageToken:
            self.metaDataParams.update({'pageToken': nextPageToken})
            url_response = json.loads(openURL("https://www.googleapis.com/youtube/v3/videos", self.metaDataParams))
            nextPageToken = url_response.get("nextPageToken")
            self.load_metadata(url_response)
        self.create_final_df()

    def get_video_replies(self, parentid):
        self.repliesParams = {
            'part': 'snippet',
            'maxResults': 100,
            'parentId': parentid,
            'textFormat': 'plainText',
            'key': self.key
        }
        url_response = json.loads(
            openURL("https://www.googleapis.com/youtube/v3/comments", self.repliesParams))
        nextPageToken = url_response.get("nextPageToken")
        self.load_replies(url_response)

        while nextPageToken:
            self.metaDataParams.update({'pageToken': nextPageToken})
            url_response = json.loads(
                openURL("https://www.googleapis.com/youtube/v3/comments", self.repliesParams))
            nextPageToken = url_response.get("nextPageToken")
            self.load_replies(url_response)

    def load_replies(self, mat):
        for item in mat["items"]:
            reply = item["snippet"]
            self.replies["parentId"].append(reply["parentId"])
            self.replies["authorDisplayName"].append(reply['authorDisplayName'])
            self.replies["replyComment"].append(reply["textDisplay"])
            self.replies["publishedAt"].append(reply["publishedAt"])
            self.replies["likeCount"].append(reply["likeCount"])

    def create_final_df(self):

        df = pd.DataFrame().from_dict(self.metadatas)
        loc = 'temp/data.json'
        with open(loc, 'w', encoding='utf-8') as file:
            df.to_json(file, orient='records', force_ascii=False)
        # df.to_json("output/" + "test.json", orient='records', force_ascii=False)
