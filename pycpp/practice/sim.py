import simpy
import random
import pandas as pd
import matplotlib.pyplot as plt
num_body = 2
mean_body = 1
std_body = 0.1
num_neck = 1
mean_neck = 1
std_neck = 0.2
num_paint = 1
mean_paint = 4
std_paint = 0.3
num_ensam = 4
mean_ensam = 1
std_ensam = 0.2
guitars_made = 0
wood_capacity = 1000
initial_wood = 100
dispatch_capacity = 500
electronic_capacity=100
initial_electronic=100
# pre_paint_capacity=100
# post_paint_capacity=200

body_pre_paint_capacity = 60
neck_pre_paint_capacity = 60
body_post_paint_capacity = 120
neck_post_paint_capacity = 120

class Guitar_Factory:
    def __init__(self,env):
        self.wood = simpy.Container(env,capacity=wood_capacity,init=initial_wood)
        self.wood_control=env.process(self.wood_stock_control(env))
        self.dispatch=simpy.Container(env,capacity=dispatch_capacity,init=0)
        self.dispatch_control = env.process(self.dispatch_guitars_control(env))
        self.electronic=simpy.Container(env,capacity=electronic_capacity,init=initial_electronic)
        self.electronic_control = env.process(self.electronic_stock_control(env))
        self.body_pre_paint = simpy.Container(env, capacity=body_pre_paint_capacity, init=0)
        self.neck_pre_paint = simpy.Container(env, capacity=neck_pre_paint_capacity, init=0)
        self.body_post_paint = simpy.Container(env, capacity=body_post_paint_capacity, init=0)
        self.neck_post_paint = simpy.Container(env, capacity=neck_post_paint_capacity, init=0)
        # self.pre_paint=simpy.Container(env,capacity=pre_paint_capacity,init=0)
        # self.post_paint=simpy.Container(env,capacity=post_paint_capacity,init=0)
        self.wood_critial_stock = (((8 / mean_body) * num_body + (8 / mean_neck) * num_neck) * 3)
        self.electronic_critial_stock = (8 / mean_ensam) * num_ensam * 2
        #self.env_status_monitor = env.process(self.env_status(env))
    def body_maker(self,env):
        #for i in range(num_body):
            while True:
                yield self.wood.get(1)
                body_time=random.gauss(mean_body,std_body)
                yield env.timeout(body_time)
                yield self.body_pre_paint.put(1)
            #yield env.timeout(0)
    def neck_maker(self,env):
        #for i in range(num_neck):
            while True:
                yield self.wood.get(1)
                neck_time=random.gauss(mean_neck,std_neck)
                yield env.timeout(neck_time)
                yield self.neck_pre_paint.put(2)
            #yield env.timeout(0)
    def painter(self,env):
        #for i in range(num_paint):
            while True:
                yield guitar_factory.body_pre_paint.get(5)
                yield guitar_factory.neck_pre_paint.get(5)
                paint_time=random.gauss(mean_paint,std_paint)
                yield env.timeout(paint_time)
                yield guitar_factory.body_post_paint.put(5)
                yield guitar_factory.neck_post_paint.put(5)
            #yield env.timeout(0)
    def assembler(self,env):
        #for i in range(num_ensam):
            while True:
                yield guitar_factory.body_post_paint.get(1)
                yield guitar_factory.neck_post_paint.get(1)
                yield self.electronic.get(1)
                assembler_time=max(random.gauss(mean_ensam,std_ensam),1)
                yield env.timeout(assembler_time)
                yield self.dispatch.put(1)
            #yield env.timeout(0)
    def wood_stock_control(self,env):
        yield env.timeout(0)
        while True:
            if self.wood.level <= self.wood_critial_stock:
                print('在第{0}日 第{1}小时，木材库存 ({2})低于预警库存水平下 '.format(int(env.now / 8), env.now % 8, self.wood.level))
                print('联系供应商')
                print('----------------------------------')
                yield env.timeout(16)
                print('在第{0}天 第{1}小时，木材送达'.format(int(env.now / 8), env.now % 8))
                yield self.wood.put(300)
                print('当前库存是：{0}'.format(self.wood.level))
                print('----------------------------------')
                yield env.timeout(8)
            else:
                yield env.timeout(1)
    def electronic_stock_control(self,env):
        yield env.timeout(0)
        while True:
            if self.electronic.level <= self.electronic_critial_stock:
                print('在第{0}日 第{1}小时，电子元件库存 ({2})低于预警库存水平下'.format(int(env.now / 8), env.now % 8, self.electronic.level))
                print('联系供应商')
                print('----------------------------------')
                yield env.timeout(16)
                print('在第{0}天 第{1}小时，电子元件送达'.format(int(env.now / 8), env.now % 8))
                yield self.electronic.put(100)
                print('当前库存是：{0}'.format(self.electronic.level))
                print('----------------------------------')
                yield env.timeout(8)
            else:
                yield env.timeout(1)
    def dispatch_guitars_control(self, env):
        global guitars_made
        yield env.timeout(0)
        while True:
            if self.dispatch.level >= 50:
                print('成品库存为：{0}, 在第{1}日 第{2}小时 联系了商场取货'.format(self.dispatch.level, int(env.now / 8), env.now % 8))
                print('----------------------------------')
                yield env.timeout(4)
                print('在第{0}日 第{1}小时，商场取走{2}吉他'.format(int(env.now / 8), env.now % 8, self.dispatch.level))
                guitars_made += self.dispatch.level
                yield self.dispatch.get(self.dispatch.level)
                print('----------------------------------')
                yield env.timeout(8)
            else:
                yield env.timeout(1)

    def env_status(self, env):
        global status
        status = pd.DataFrame(
            columns=["datetime", "dispatch_level", 'wood', 'electronic', 'body_pre_paint', 'neck_pre_paint',
                     'body_post_paint', 'neck_post_paint'])
        status[
            ["datetime", "dispatch_level", 'wood', 'electronic', 'body_pre_paint', 'neck_pre_paint', 'body_post_paint',
             'neck_post_paint']] = status[
            ["datetime", "dispatch_level", 'wood', 'electronic', 'body_pre_paint', 'neck_pre_paint', 'body_post_paint',
             'neck_post_paint']].astype(int)
        while True:
            im = plt.plot(status['datetime'], status['neck_pre_paint'], color='#4D9221')
            im.append(im)
            plt.title('neck_pre_paint')
            plt.pause(0.001)
            print('{0}在第{1}日 第{2}小时，成品库存量：{3}'.format(env.now, int(env.now / 8), env.now % 8, self.dispatch.level))
            tmp = {'datetime': env.now,
                   'dispatch_level': self.dispatch.level,
                   'wood': self.wood.level,
                   'electronic': self.electronic.level,
                   'body_pre_paint': self.body_pre_paint.level,
                   'neck_pre_paint': self.neck_pre_paint.level,
                   'body_post_paint': self.body_post_paint.level,
                   'neck_post_paint': self.neck_post_paint.level
                   }
            status = status.append([tmp])
            yield env.timeout(1)

def body_maker_gen(env, guitar_factory):
    for i in range(num_body):
        env.process(guitar_factory.body_maker(env))
        yield env.timeout(0)
def neck_maker_gen(env, guitar_factory):
    for i in range(num_neck):
        env.process(guitar_factory.neck_maker(env))
        yield env.timeout(0)
def paint_maker_gen(env, guitar_factory):
    for i in range(num_paint):
        env.process(guitar_factory.painter(env))
        yield env.timeout(0)
def assembler_maker_gen(env, guitar_factory):
    for i in range(num_ensam):
        env.process(guitar_factory.assembler(env))
        yield env.timeout(0)
hours=8
days=5
total_time=hours*days

env=simpy.Environment()
guitar_factory=Guitar_Factory(env)
body_gen = env.process(body_maker_gen(env, guitar_factory))
#body_maker_process=env.process(guitar_factory.body_maker(env))
neck_gen = env.process(neck_maker_gen(env, guitar_factory))
#neck_maker_process = env.process(guitar_factory.neck_maker(env))
paint_gen = env.process(paint_maker_gen(env, guitar_factory))
#painter_process=env.process(guitar_factory.painter(env))
assembler_gen = env.process(assembler_maker_gen(env, guitar_factory))
#assembler_process=env.process(guitar_factory.assembler(env))

a=env.process(guitar_factory.env_status(env))

print('仿真开始：')
env.run(until = total_time)
print('当前等待喷涂的琴身数量：{0} 和琴颈数量： {1}'.format(guitar_factory.body_pre_paint.level, guitar_factory.neck_pre_paint.level))
print('当前等待组装的琴身数量：{0} 和琴颈数量： {1}'.format(guitar_factory.body_post_paint.level, guitar_factory.neck_post_paint.level))
print(f'当前成品库存量： %d ' % guitar_factory.dispatch.level)
print(f'----------------------------------')
print('此周期的吉他总生产数量: {0}'.format(guitars_made + guitar_factory.dispatch.level))
print(f'----------------------------------')
print(f'仿真完成！')
