# SPDX-License-Identifier: LGPL-3.0-or-later
import itertools

import numpy as np
from tqdm.auto import tqdm

from .utils import SharedRNGData, bytestolist, listtobytes, read_compressed_block


class _mergeISO(SharedRNGData):
    miso: int
    moleculetempfilename: str
    temp1it: int

    def __init__(self, rng):
        SharedRNGData.__init__(self, rng, ["miso", "moleculetempfilename"], ["temp1it"])

    def merge(self):
        if self.miso > 0:
            self._mergeISO()
        self.returnkeys()

    def _mergeISO(self):
        items = []
        with open(self.moleculetempfilename, "rb") as ft:
            # r=读，b=二进制
            # print('moleculetempfilename:', self.moleculetempfilename)
            for item in itertools.zip_longest(*[read_compressed_block(ft)] * 4):
                items.append(item)
            # item是一个以二进制为元素的列表，其中包含的是每一个识别出来的分子的信息
            # 包括原子序数，键连关系，键级
            # print('items:', items)
        
        new_items = []
        _oldbitem = b"" # b代表二进制数据
        _oldbbond = b"0" # b代表二进制数据
        _oldindex = []
        _oldfreq = 0
        
        # 新建测试文件，存储_bitem, _bbond0, _bbond1, _bindex
        _testfile = open('test.txt', 'w')

        # tqdm显示任务进度
        for _bitem, _bbond0, _bbond1, _bindex in tqdm(
            sorted(items, key=lambda x: (x[0], x[1])), # 对items进行排序，按照分子的大小排序预处理，并不涉及融合异构体的过程
            desc="Merge isomers:", # 进度名称
            disable=None, # 根据环境为交互式/非交互式，显示/不显示进度条
        ):
            print('_bitem', bytestolist(_bitem), len(bytestolist(_bitem)),    # 原子编号 
                  '_bbond0', bytestolist(_bbond0),  # 键连信息
                  '_bbond1', bytestolist(_bbond1),  # 键级
                  '_bindex', bytestolist(_bindex),  # 帧的编号，即在时间尺度上是否存在
                  sep = '\n',
                  file = _testfile)
            
            _index = bytestolist(_bindex)
            #### 打印_index
            # print('_index:', _index)
            
            _freq = len(_index)
            #### 打印_freq
            # print('_freq:', _freq)
            
            # miso=1时，第一轮循环运行的是else
            # 第二轮循环起，如果当前分子与原先分子的原子编号相同且键连关系相同，那么合并两分子的时间帧编号
            if (_bitem == _oldbitem) and ((_bbond0 == _oldbbond[0]) or (self.miso > 1)):
                _oldindex = np.hstack((_oldindex, _index)) # 水平合并两个列表，生成np数组
                #### 打印 _oldindex
                # print('if _oldindex', _oldindex)

            else:
                if _oldbitem: # 如果_oldbitem不为空，第一轮循环_olditem是空的，直到不一样的分子出现
                    # print(bytestolist(_oldbitem), bytestolist(_oldbbond[0]), bytestolist(_oldbbond[1]))
                    new_items.append([_oldbitem, *_oldbbond, listtobytes(_oldindex)])
                    # 将前面符合融合规则的分子信息写入新的列表中，*代表将键连信息和键级拆包
                    # 由于预先进行过排序，因此后续不应该有可以和该分子融合的分子
                    # 融合后，写入的信息是排序是位列这一种分子第一个的分子的原子编号、键连信息、键级

                    # 此处重置_oldfreq
                    _oldfreq = 0
                    #### 2024/04/02 ####

                # 将新分子的信息赋值，以便进行比对
                _oldbitem = _bitem
                _oldindex = _index

            # 此处减少一个缩进 # 
            #### 2024/04/02 ####

            if _freq > _oldfreq: # 如果_freq > 0，即_index不为空
                
                #### 此处添加更新oldfreq ####
                _oldfreq = _freq
                #### 2024/04/02 ####

                _oldbbond = (_bbond0, _bbond1)
                # 第一轮过后，原本空的_oldbitem, _oldbbond, _oldindex
                # 都变为items排序后第一个分子的_bitem, (_bbond0, _bbond1), _index
                # 原本_oldfreq始终为0

        _testfile.close()
        
        new_items.append([_oldbitem, *_oldbbond, listtobytes(_oldindex)])
        # 写入最后一个符合融合规则的分子信息
        new_items.sort(key=lambda x: len(x[0]))
        # 将new.items按分子大小排序
        self.temp1it = len(new_items)
        with open(self.moleculetempfilename, "wb") as ft:
            for item in new_items:
                [ft.write(i) for i in item]
        
