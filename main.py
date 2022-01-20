import math
import random
import pygame
import os
import numpy as np

pygame.init()
W = 800
H = 800

sc = pygame.display.set_mode((W, H))
pygame.display.set_caption("CompGr_Lab3_3")

WHITE = (255, 255, 255)
GREY = (200, 200, 200)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

FPS = 60  # число кадров в секунду
clock = pygame.time.Clock()


class InputRect:
    base_font = pygame.font.Font(None, 32)
    text = ''
    col_act = pygame.Color('lightskyblue3')
    col_pass = pygame.Color('chartreuse4')
    col = col_pass
    active = False
    rect = 0
    action = lambda x: 0

    def __init__(self, pos, action):
        self.rect = pygame.Rect(pos[0], pos[1], 140, 32)
        self.action = action

    def check_color(self):
        if self.active:
            self.col = self.col_act
        else:
            self.col = self.col_pass

    def draw(self):
        self.check_color()
        pygame.draw.rect(sc, self.col, self.rect)
        text_surface = self.base_font.render(self.text, True, (255, 255, 255))
        sc.blit(text_surface, (self.rect.x + 5, self.rect.y + 5))
        self.rect.w = max(100, text_surface.get_width() + 10)

    def apply(self, p):
        res = p
        str_args = self.text.split()
        if self.action == translation:
            if len(str_args) == 2:
                dp = self.text.split()
                dp = np.array([int(dp[0]), int(dp[1]), 0])
                # dp = np.array([10, 10, 0])
                res = translation(p, dp)
        elif self.action == rotation:
            if len(str_args) == 1:
                res = rotation(p, float(str_args[0])*math.pi/180)
            elif len(str_args) == 3:
                a, c = float(str_args[0])*math.pi/180, np.array([int(str_args[1]), int(str_args[2]), 0])
                res = rotation(p, a, c)
        elif self.action == scaling:
            if len(str_args) == 2:
                res = scaling(p, np.array([float(str_args[0]), float(str_args[1])]))
            elif len(str_args) == 4:
                s, c = np.array([float(str_args[0]), float(str_args[1])]), \
                       np.array([int(str_args[2]), int(str_args[3])])
                res = scaling(p, s, c)
        elif self.action == shear:
            if len(str_args) == 1:
                res = shear(p, float(str_args[0])*math.pi/180)
            elif len(str_args) == 3:
                a, c = float(str_args[0])*math.pi/180, np.array([int(str_args[1]), int(str_args[2]), 0])
                res = shear(p, a, c)
            elif len(str_args) == 5:
                a, c, axis = float(str_args[0])*math.pi/180, np.array([int(str_args[1]), int(str_args[2]), 0]), \
                             np.array([float(str_args[3]), float(str_args[4])])
                res = shear(p, a, c, axis)
        return res


def tp(p):
    return np.array([W // 2 + p[0], H // 2 - p[1]])


def td(p):
    return p[0] - W // 2, H // 2 - p[1], 0


def translation(p, dp):
    # print(f"Translation. dp = {dp[0]}, {dp[1]}")
    m = np.array([[1, 0, dp[0]],
                  [0, 1, dp[1]],
                  [0, 0, 1]])
    return m @ p


def rotation(p, a, c=np.array([0, 0, 0])):
    # print(f"Rotation. angle = {a}, center = ({c[0]}, {c[1]})")
    m = np.array([[np.cos(a), -np.sin(a), 0],
                  [np.sin(a), np.cos(a), 0],
                  [0, 0, 1]])
    return translation(m @ translation(p, -c), c)


def scaling(p, s, c=np.array([0, 0, 0])):
    # print(f"Scaling. scale = ({s[0]}, {s[1]}), center = ({c[0]}, {c[1]})")
    m = np.array([[s[0], 0, 0],
                  [0, s[1], 0],
                  [0, 0, 1]])
    return translation(m @ translation(p, -c), c)


def shear(p, a, c=np.array([0, 0, 0]), axis=np.array([0, 1])):
    # print(f"Shearing. angle = {a}, center = ({c[0]}, {c[1]}), axis = ({axis[0]}, {axis[1]})")
    cos_ax_ang = axis[1] / np.sqrt(axis[0] * axis[0] + axis[1] * axis[1])
    ang = np.arccos(cos_ax_ang)
    m = np.array([[1, np.tan(a), 0],
                  [0, 1, 0],
                  [0, 0, 1]])
    return translation(rotation(m @ rotation(translation(p, -c), ang), -ang), c)


def rand_poly():
    M = random.randint(1, 6) % 6 + 3
    res = np.zeros((M, 3))
    angs = np.sort(np.random.rand(M) * 2 * math.pi)
    rads = (0.5 + 0.5 * np.sqrt(np.random.rand(M))) * min(W // 8, H // 8)
    x_r = random.randint(3 * W // 8, 5 * W // 8) - W / 2
    y_r = random.randint(3 * H // 8, 5 * H // 8) - H / 2
    for i in range(M):
        res[i][0] = int(x_r + rads[i] * np.cos(angs[i]))
        res[i][1] = int(y_r + rads[i] * np.sin(angs[i]))
        res[i][2] = 1
    return res


def get_c_m(points):
    if points.shape[0] == 3:
        return np.mean(points, axis=0)
    z = np.mean(points, axis=0)
    s = np.cross(points[-1]-z, points[0]-z)[2]/2
    s_r = get_c_m(np.array([z, points[0], points[-1]]))*s
    for i in range(points.shape[0]-1):
        s_i = np.cross(points[i]-z, points[i+1]-z)[2]/2
        s += s_i
        s_r += get_c_m(np.array([z, points[i], points[i+1]]))*s_i
    res = s_r / s
    return res


poly = rand_poly()

inputs = [InputRect((0, 20), translation), InputRect((0, 70), rotation),
          InputRect((0, 120), scaling), InputRect((0, 170), shear)]

anim = 0
anim_1 = 1
anim_2 = 1
anim_3 = 1
anim_4 = 1

while 1:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            exit()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_F1:
                anim = 1 * (anim != 1)
                anim_1 = -anim_1 if (anim != 1) else anim_1
                continue
            elif event.key == pygame.K_F2:
                anim = 2 * (anim != 2)
                anim_2 = -anim_2 if (anim != 2) else anim_2
                continue
            elif event.key == pygame.K_F3:
                anim = 3 * (anim != 3)
                anim_3 = -anim_3 if (anim != 3) else anim_3
                continue
            elif event.key == pygame.K_F4:
                anim = 4 * (anim != 4)
                anim_4 = -anim_4 if (anim != 4) else anim_4
                continue
        if anim != 0:
            continue
        if event.type == pygame.MOUSEBUTTONDOWN:
            for inp in inputs:
                if inp.rect.collidepoint(event.pos):
                    inp.active = True
                else:
                    inp.active = False
        if event.type == pygame.KEYDOWN:
            for inp in inputs:
                if inp.active:
                    if event.key == pygame.K_BACKSPACE:
                        inp.text = inp.text[:-1]
                    elif event.key == pygame.K_RETURN:
                        inp.active = False
                        for i in range(poly.shape[0]):
                            poly[i] = inp.apply(poly[i])
                        inp.text = ""
                    elif event.key != pygame.K_F5:
                        inp.text += event.unicode
            if event.key == pygame.K_F5:
                poly = rand_poly()
            inactive = True
            for inp in inputs:
                inactive = inactive and not inp.active
            if inactive and event.key == pygame.K_SPACE:
                c_m = get_c_m(poly)
                for i in range(poly.shape[0]):
                    poly[i] = translation(poly[i], -c_m)

    sc.fill(WHITE)

    # net
    vert = np.append(np.arange(W / 2, 0, -20), np.arange(W / 2, W, 20))
    horiz = np.append(np.arange(H / 2, 0, -20), np.arange(H / 2, H, 20))
    for v_line in vert:
        pygame.draw.line(sc, GREY, (v_line, 0), (v_line, H))
    for h_line in horiz:
        pygame.draw.line(sc, GREY, (0, h_line), (W, h_line))
    pygame.draw.line(sc, BLACK, (W / 2, 0), (W / 2, H))
    pygame.draw.line(sc, BLACK, (0, H / 2), (W, H / 2))

    # inputs
    for inp in inputs:
        inp.draw()

    # anim
    if anim == 1:
        for i in range(poly.shape[0]):
            poly[i] = translation(poly[i], [anim_1*2, anim_1*1])
    elif anim == 2:
        c_m = get_c_m(poly)
        for i in range(poly.shape[0]):
            poly[i] = rotation(poly[i], 0.01 * anim_2, c_m)
    elif anim == 3:
        c_m = get_c_m(poly)
        for i in range(poly.shape[0]):
            poly[i] = scaling(poly[i], (1.01 ** anim_3, 1.01 ** anim_3), c_m)
    elif anim == 4:
        c_m = get_c_m(poly)
        for i in range(poly.shape[0]):
            poly[i] = shear(poly[i], 0.01 * anim_4, c_m, np.array([1, -1]))

    # poly
    pygame.draw.polygon(sc, BLUE, [tp(p[0:2]) for p in poly])
    pygame.display.update()
    clock.tick(FPS)
