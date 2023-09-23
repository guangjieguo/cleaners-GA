__author__ = "Lech Szymanski"
__organization__ = "COSC343/AIML402, University of Otago"
__email__ = "lech.szymanski@otago.ac.nz"

import pygame
import numpy as np
import sys

class visualiser:

   def __init__(self, gridSize, speed, resolution=(720,480), playerStrings=None):
      pygame.init()

      self.playerStrings = playerStrings

      self.width, self.height = resolution
      self.WHITE = 255, 255, 255
      self.BLACK = 0, 0, 0
      self.GREY = 128, 128, 128
      self.YELLOW = 250, 251, 0
      self.BLUE = 0,155,255
      self.MAGENTA = 255, 64, 255
      self.RED = 255,38,0
      self.GREEN = 155,225,0

      if speed == "normal":
         self.frameTurns = 400
      elif speed == "fast":
         self.frameTurns = 100
      elif speed == "slow":
         self.frameTurns = 1000

      self.screen = pygame.display.set_mode(resolution)

      self.font = pygame.font.Font("arial.ttf", 14)

      minMargin = 200

      self.Y, self.X = gridSize

      unitX = (self.width - 2*minMargin)/self.X
      unitY = self.height / self.Y

      self.unit = np.min([unitX, unitY])
      if self.unit < 1:
          self.unit = 1

      self.marginX = (self.width - self.unit*self.X)/2
      self.marginY = (self.height - self.unit*self.Y)/2

      self.cleaner_size = self.unit/2 * 0.75

   def __del__(self):
      pygame.display.quit()
      pygame.quit()

   def show(self, vis_data, turn=0, game=None, titleStr = None):
       map, cleaners, stats = vis_data

       if titleStr is None:
           caption = ''
       else:
           caption = titleStr + ', '

       if game is not None:
           if isinstance(game, str):
               caption += 'Game %s ' % game
           else:
               caption += 'Game %d' % game
               if turn > 0:
                   caption += ", "

       if turn > 0:
           caption += 'Turn %d' % (turn)

       pygame.display.set_caption(caption)



       for event in pygame.event.get():
           if event.type == pygame.QUIT: sys.exit()

       self.screen.fill(self.WHITE)

       if self.playerStrings is not None:
           label = self.font.render(self.playerStrings[0], 1, self.BLUE)
           self.screen.blit(label, (self.marginX - 105, self.marginY + 10))
           label = self.font.render("cleaned: %d" % stats['cleaned'][0], 1, self.BLUE)
           self.screen.blit(label, (self.marginX - 100, self.marginY + 30))

           if len(self.playerStrings) > 1:
               label = self.font.render(self.playerStrings[1], 1, self.MAGENTA)
               self.screen.blit(label, (self.marginX + (self.X * self.unit) + 10, self.marginY + 10))
               label = self.font.render("cleaned: %d" % stats['cleaned'][1], 1, self.MAGENTA)
               self.screen.blit(label, (self.marginX + (self.X * self.unit) + 15, self.marginY + 30))


       for y in range(self.Y):
           for x in range(self.X):
                if map[y,x] == 10:
                    c = self.BLACK
                elif map[y,x] == -1:
                    c = self.GREY
                elif map[y,x] == 1:
                    c = self.GREEN
                else:
                    continue

                pygame.draw.rect(self.screen, c,
                                 (self.marginX + (x * self.unit), self.marginY + y * self.unit, np.ceil(self.unit), np.ceil(self.unit)))

       for y,x,r,e,b,p in cleaners:

           if p==0:
               c = self.BLUE
           else:
               c = self.MAGENTA

           centre_offset = self.unit/2

           pygame.draw.circle(self.screen,c,(self.marginX + x * self.unit + centre_offset,self.marginY + y * self.unit + centre_offset ),self.cleaner_size)

           if e < 0.33:
             c = self.RED
           elif e < 0.66:
             c = self.YELLOW
           else:
             c = self.GREEN

           if r==0:
               xo = 0
               yo = -1
           elif r==90:
               xo = 1
               yo = 0
           elif r==180:
               xo = 0
               yo = 1
           else:
               xo = -1
               yo = 0


           radius = self.cleaner_size / 3
           if radius < 1:
               radius = 1
           offset = self.cleaner_size * 0.6
           pygame.draw.circle(self.screen,c,(self.marginX + x * self.unit + centre_offset + xo*offset,self.marginY + y * self.unit + centre_offset + yo * offset),radius)

           if r==0:
               xb = [-1, 0, 1]
               yb = [0, 0, 0]
           elif r==90:
               xb = [0, 0, 0]
               yb = [1, 0, -1]
           elif r==180:
               xb = [1, 0, -1]
               yb = [0, 0, 0]
           else:
               xb = [0, 0, 0]
               yb = [-1, 0, 1]


           b = int(b*3)
           boff = radius
           for i in range(b):
               pygame.draw.circle(self.screen, self.GREY, (self.marginX + x * self.unit + centre_offset - xo * offset + boff * xb[i],
                                                   self.marginY + y * self.unit + centre_offset - yo * offset + boff * yb[i]), radius/2)

       for y in range(self.Y+1):
            pygame.draw.line(self.screen, self.BLACK, [self.marginX, self.marginY + y * self.unit],
                             [self.marginX + (self.X * self.unit), self.marginY +  y * self.unit])
       for x in range(self.X+1):
            pygame.draw.line(self.screen, self.BLACK, [self.marginX + (x * self.unit), self.marginY],
                             [self.marginX + (x * self.unit), self.marginY + self.Y * self.unit])

       pygame.display.flip()

       pygame.time.delay(self.frameTurns)


