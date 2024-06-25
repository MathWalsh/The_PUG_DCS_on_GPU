#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


SlackChatBot

class so that status / error messages from a python app can be sent to a slack channel

You to create an app on the slack API web site, and create / retreive two tokens:
    
    slack_bot_token  beginning with  "xoxb-"
    slack_app_token  beginning with  "xapp-"
    
    In "OAuth & Permissions" make sure that you allow:
            channels:join
            channels:read
            chat:write
            channels:read
            
    The bot is initialized with your tokens, a name and the channel you want it to send messages
    
    Channel is joined by uisng 
        join_channel
        
    Messages are sent with
        send_message
        send_messageToChannel      // Allows sending to a different specific channel
    
    Messages will appear in the channel will appear as "Bot Name: This is my message""
                
    In a future version, we want to add the ability to send control messages
    

Use case examples are provided in the main() testing function below


@author: jgenest
"""


from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from slack_sdk.socket_mode import SocketModeClient
from slack_sdk.socket_mode.request import SocketModeRequest
from slack_sdk.socket_mode.response import SocketModeResponse

import threading
from queue import Queue

import time

"""
For future use, when SlacBot will be able to receive commads

template for the responder class, implement your own elsewhere
"""


class Responder:
    def newMessageSlack(self, message):
        """
        Handles a new message from slack, responder function
        """
        print(f"{self.name} Responder received: {message}")
        
    def errorFromSlack(self, error):
        """
        Handles an sack error.
        """
        print(f"UDP error: {error}")
    


class SlackChatBot:
    def __init__(self, bot_token, app_token,name="Default Bot", channel="" , start_delimiter="<<", stop_delimiter=">>"):
        self.client = WebClient(token=bot_token)
        self.socket_mode_client = SocketModeClient(app_token=app_token,web_client=self.client)
        
        self.name = name
        self.channel = channel
        self.responder = None
        self.startDelimiter = start_delimiter
        self.stopDelimiter = stop_delimiter
        
        self.last_message = ""
        self.last_message_time = 0
        self.timeout = 60  # Timeout period in seconds
        
        self.running = False
        self.message_queue = Queue()
        
    def __del__(self):
       self.running = False
       self.message_queue.put((None, None))
       self.thread.join()   
        
    def _threadFunction(self):
        while self.running:
            print('yes')
            channel, text = self.message_queue.get()
            if channel is None:
                break
            self._send_messageToChannel(channel, text)
            
    def _send_messageToChannel(self, channel, text):
      try:
          message = self.name + ":  " + text
          
          current_time = time.time()

          # Check if the message is the same as the last one sent within the timeout period
          if message == self.last_message and (current_time - self.last_message_time) < self.timeout:
              #print("Message not sent: Duplicate message within timeout period.")
              return
          
          # Send the message
          response = self.client.chat_postMessage(channel=channel, text=message)
          
          # Update the last message and time
          self.last_message = message
          self.last_message_time = current_time

      except SlackApiError as e:
          self.handleError(f"Error sending message: {e.response['error']}")    


    def setResponder(self, responder):
        """
        Sets the responder object which has methods to handle new messages (parser) and errors.
        """
        self.responder = responder

    def setStartDelimiter(self, delimiter):
        """
        Sets the start delimiter used to identify the beginning of a message.
        """
        self.startDelimiter = delimiter

    def setStopDelimiter(self, delimiter):
        """
        Sets the stop delimiter used to identify the end of a message.
        """
        self.stopDelimiter = delimiter


    def send_message(self, text):
        self.send_messageToChannel(self.channel,text)
        
    def send_messageToChannel(self, channel, text):
        self.message_queue.put((channel, text))

    def process_message(self, data):
        if 'text' in data and 'channel' in data:
            message = data['text']
            channel = data['channel']

            # Check if the message contains the bot's name
            if self.name in message and self.channel in channel:
                self.buffer += message

                startIndex = self.buffer.find(self.startDelimiter)
                stopIndex = self.buffer.find(self.stopDelimiter, startIndex + len(self.startDelimiter))

                if startIndex != -1 and stopIndex != -1:
                    # Extract the message between the delimiters
                    #message_with_delimiters = self.buffer[startIndex:stopIndex + len(self.stopDelimiter)]
                    message_no_delimiters = self.buffer[startIndex + len(self.startDelimiter):stopIndex]

                    # Remove the processed message from the buffer
                    self.buffer = self.buffer[stopIndex + len(self.stopDelimiter):]

                    # Pass the message without delimiters to the responder
                    self.responder.newMessageSlack(message_no_delimiters)


    def handleError(self, err=None):
        """
        Error handling is passed to the responder class and/or on a QT error signal.
    
        Parameters:
            err (str, optional): The error message to handle. Defaults to None.
            """
        
        err = str(err) if err is not None else 'Unknown error'
        if self.responder:
            self.responder.errorFromSlack(err)
        else:
            print(err)


    def join_channel(self, channel):
        try:
            response = self.client.conversations_join(channel=channel)

        except SlackApiError as e:
            self.handleError(f"Error joining channel: {e.response['error']}")

    def start(self):
        @self.socket_mode_client.socket_mode_request_listeners.append
        def handle(client: SocketModeClient, req: SocketModeRequest):
            if req.type == "events_api":
                event = req.payload["event"]
                if 'bot_id' not in event:  # Ignore messages from other bots
                    self.process_message(event)
                response = SocketModeResponse(envelope_id=req.envelope_id)
                client.send_socket_mode_response(response)

        self.socket_mode_client.connect()
        
        self.running = True;
        self.thread = threading.Thread(target=self._threadFunction)
        self.thread.start()
        
 

if __name__ == "__main__":
    
    # Go to the slack api website and install an app
    # In "OAuth & Permissions" make sure that you allow:
    #            channels:join
    #            channels:read
    #            chat:write
    #            channels:read
    # tkae note of the bot and app tokens (startitng with xoxb-  and xapp- respectively
    # In slack, right clik on the channel you want the bot to talk to, copy its link, the last part is the chanel ID
    
                                         
    slack_bot_token = "xoxb-"     # your token number from the slack api web site
    slack_app_token = "xapp-"
    
    bot_name = "TestBot"
    channelID = "C077..."

    


    resp = Responder()    
    bot = SlackChatBot(slack_bot_token, slack_app_token,bot_name, channelID)
    bot.join_channel(channelID) 
    bot.setResponder(resp)
    bot.start()
    
    bot.send_messageToChannel(channelID, "Hello, sending to specific channel")
    bot.send_message("Joining the channel")
  