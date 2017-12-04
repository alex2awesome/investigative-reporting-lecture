#!/usr/bin/env python

import tornado.ioloop
from socket import gethostname
from tornado.ioloop import IOLoop
from tornado.options import define, options, parse_command_line
from tornado.web import Application, RequestHandler, StaticFileHandler
from tornado.websocket import WebSocketHandler

# Command line options. Use --help to list
define('port', default=54321, help='HTTP port to listen on')
define('autoreload', default=False, help='Automatically watch for Python changes and restart webserver')
define('staticfiles', default='static', help='Path to static files that shall get served at /static')

# State
current_slide = 0

# Who needs to be notified when the slide changes.
slide_change_listeners = set()


class SlidesControlHandler(RequestHandler):
  """Any HTTP request to /slides-control shall increment the current slide
  number and notify all listener."""

  def get(self):
    global current_slide
    current_slide += 1
    print('Slide updated: %d' % current_slide)
    for listener in slide_change_listeners:
      listener(current_slide)
    self.write(str(current_slide))


class SlidesSubscribeHandler(WebSocketHandler):
  """WebSocket endpoint at /slides-subscribe. Upon connection it shall
  send a message saying what the current slide is, and then send a new
  message whenever the slide changes.
  The protocol is simply a number encoded as a string. e.g. '4'. """

  def open(self):
    print('Websocket opened')
    self.write_message(str(current_slide))
    slide_change_listeners.add(self.on_slide_change)

  def on_close(self):
    print('Websocket closed')
    slide_change_listeners.remove(self.on_slide_change)

  def on_slide_change(self, slide_number):
    self.write_message(str(slide_number))


def main():
  parse_command_line()
  application = Application([
    (r"/slides-control", SlidesControlHandler),
    (r"/slides-subscribe", SlidesSubscribeHandler),
    (r"/()", StaticFileHandler, dict(path='2017-12-01__demo.slides.html')),
  ], debug=options.autoreload, static_path=options.staticfiles)
  application.listen(options.port)
  print('Listening on http://%s:%s/' % ('localhost', options.port))
  IOLoop.instance().start()


if __name__ == '__main__':
  main()