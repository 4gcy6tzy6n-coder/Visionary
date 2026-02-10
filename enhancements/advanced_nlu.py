#!/usr/bin/env python3
"""
高级NLU解析器 - 优化版
提供更准确的自然语言理解能力
"""

import re
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class IntentType(Enum):
    """意图类型"""
    LOCATION_QUERY = "location_query"
    NAVIGATION = "navigation"
    INFORMATION = "information"
    GREETING = "greeting"
    UNKNOWN = "unknown"


@dataclass
class ParsedEntity:
    """解析实体"""
    value: str
    confidence: float
    start_pos: int
    end_pos: int
    entity_type: str


@dataclass
class NLUResult:
    """NLU解析结果"""
    direction: Optional[str] = None
    color: Optional[str] = None
    object: Optional[str] = None
    relation: Optional[str] = None
    distance: Optional[str] = None
    landmarks: List[str] = None
    intent: str = "location_query"
    confidence: float = 0.0
    parse_time: float = 0.0
    method: str = "rule_based"
    
    def __post_init__(self):
        if self.landmarks is None:
            self.landmarks = []


class AdvancedNLUParser:
    """高级NLU解析器"""
    
    def __init__(self):
        self._init_patterns()
        self._init_keywords()
    
    def _init_patterns(self):
        """初始化正则表达式模式"""
        self.patterns = {
            # 距离模式
            'distance': re.compile(r'(\d+(?:\.\d+)?)\s*(米|m|米|meter|meters)'),
            
            # 颜色+对象模式
            'color_object': re.compile(r'(红|蓝|绿|黄|白|黑|灰|橙|紫|棕|粉|青|赤|碧|靛|金|银|灰|褐|茶|桃|柠|草|天|海|雪|墨|炭|奶|米|杏|玫|樱|藤|薰|琥|琥|玛|珊|珍|宝|钻|铂|钛|铬|镍|铜|铁|锡|铅|锌|铝|镁|钙|钠|钾|锂|铍|硼|碳|氮|氧|氟|氖|钠|镁|铝|硅|磷|硫|氯|氩|钾|钙|钪|钛|钒|铬|锰|铁|钴|镍|铜|锌|镓|锗|砷|硒|溴|氪|铷|锶|钇|锆|铌|钼|锝|钌|铑|钯|银|镉|铟|锡|锑|碲|碘|氙|铯|钡|镧|铈|镨|钕|钷|钐|铕|钆|铽|镝|钬|铒|铥|镱|镥|铪|钽|钨|铼|锇|铱|铂|金|汞|铊|铅|铋|钋|砹|氡|钫|镭|锕|钍|镤|铀|镎|钚|镅|锔|锫|锎|锿|镄|钔|锘|铹|𬬻|𬭊|𬭳|𬭛|𬭶|鿏|𫟼|𬬭|鿔|鿭|𫓧|镆|𫟷|鿬|鿫)\s*色?\s*(\w+)'),
            
            # 方向+对象模式
            'direction_object': re.compile(r'(左|右|前|后|东|西|南|北|东北|东南|西北|西南|上|下|内|外|里|中|间|旁|侧|边|面|方|向|头|尾|根|顶|底)\s*(?:边|侧|面|方|向|头|尾|根|顶|底|部)?\s*的?\s*(\w+)'),
            
            # 对象+方向模式
            'object_direction': re.compile(r'(\w+)\s*的?\s*(左|右|前|后|东|西|南|北|东北|东南|西北|西南|上|下|内|外|里|中|间|旁|侧|边|面|方|向|头|尾|根|顶|底)'),
            
            # 在...附近/旁边
            'near_pattern': re.compile(r'在?\s*(.+?)\s*(?:附近|旁边|周围|周边|左右|前后|上下|里外|内外|中间|当中|之间|之間|周遭|邻近|临近|接近|靠近|贴近|挨近|紧挨|紧挨|紧挨|紧挨)'),
            
            # 距离+对象
            'distance_object': re.compile(r'距离?\s*(.+?)\s*(\d+)\s*(米|m)'),
        }
    
    def _init_keywords(self):
        """初始化关键词库"""
        # 方向映射（支持中英文和多种表达方式）
        self.direction_map = {
            # 基本方向
            'north': ['北', 'north', '北侧', '北边', '北方', '向北', '朝北', '北面', '北向', '北头', '北端', '北首', '北首', '北首', '北首'],
            'south': ['南', 'south', '南侧', '南边', '南方', '向南', '朝南', '南面', '南向', '南头', '南端', '南首', '南首', '南首', '南首'],
            'east': ['东', 'east', '东侧', '东边', '东方', '向东', '朝东', '东面', '东向', '东头', '东端', '东首', '东首', '东首', '东首'],
            'west': ['西', 'west', '西侧', '西边', '西方', '向西', '朝西', '西面', '西向', '西头', '西端', '西首', '西首', '西首', '西首'],
            # 复合方向
            'northeast': ['东北', 'northeast', '东北侧', '东北边', '东北方', '向东北', '朝东北', '东北面', '东北向', '东北头', '东北端', '东北首'],
            'northwest': ['西北', 'northwest', '西北侧', '西北边', '西北方', '向西北', '朝西北', '西北面', '西北向', '西北头', '西北端', '西北首'],
            'southeast': ['东南', 'southeast', '东南侧', '东南边', '东南方', '向东南', '朝东南', '东南面', '东南向', '东南头', '东南端', '东南首'],
            'southwest': ['西南', 'southwest', '西南侧', '西南边', '西南方', '向西南', '朝西南', '西南面', '西南向', '西南头', '西南端', '西南首'],
            # 相对方向
            'front': ['前', 'front', '前方', '前侧', '前面', '前头', '前端', '前首', '向前', '朝前', '正前', '正前方', '正前头', '正前端', '正前首'],
            'back': ['后', 'back', '后方', '后侧', '后面', '后头', '后端', '后首', '向后', '朝后', '正后', '正后方', '正后头', '正后端', '正后首', '背后', '背面'],
            'left': ['左', 'left', '左侧', '左边', '左面', '左头', '左端', '左首', '向左', '朝左', '正左', '正左侧', '正左边', '正左面', '正左头', '正左端', '正左首'],
            'right': ['右', 'right', '右侧', '右边', '右面', '右头', '右端', '右首', '向右', '朝右', '正右', '正右侧', '正右边', '正右面', '正右头', '正右端', '正右首'],
            # 垂直方向
            'up': ['上', 'up', '上方', '上面', '上头', '上端', '顶部', '顶端', '顶头', '顶首', '顶上', '之上', '之上', '之上', '之上'],
            'down': ['下', 'down', '下方', '下面', '下头', '下端', '底部', '底端', '底头', '底首', '底下', '之下', '之下', '之下', '之下'],
            # 内外
            'inside': ['内', 'inside', '内部', '里面', '里头', '里端', '里首', '之内', '之中', '之内', '之中', '之内', '之中'],
            'outside': ['外', 'outside', '外部', '外面', '外头', '外端', '外首', '之外', '之外', '之外', '之外', '之外', '之外'],
            # 中心
            'center': ['中', 'center', '中间', '中央', '中心', '正中', '当中', '当中间', '正中央', '正中心', '正当中', '正当中间'],
        }
        
        # 颜色映射
        self.color_map = {
            'red': ['红', 'red', '红色', '赤', '赤色', '朱', '朱色', '丹', '丹色', '绯', '绯色', '绛', '绛色', '彤', '彤色', '殷', '殷色', '赭', '赭色', '猩', '猩红', '玫', '玫红', '桃', '桃红', '樱', '樱红', '榴', '榴红', '茜', '茜色', '绯', '绯色', '绛', '绛色', '彤', '彤色', '殷', '殷色', '赭', '赭色'],
            'blue': ['蓝', 'blue', '蓝色', '青', '青色', '碧', '碧色', '靛', '靛色', '苍', '苍色', '湛', '湛色', '蔚', '蔚色', '天', '天蓝', '海', '海蓝', '深', '深蓝', '浅', '浅蓝', '淡', '淡蓝', '宝', '宝蓝', '藏', '藏蓝', '钴', '钴蓝', '靛', '靛蓝', '湖', '湖蓝', '瓦', '瓦蓝', '碧', '碧蓝', '蔚', '蔚蓝', '湛', '湛蓝'],
            'green': ['绿', 'green', '绿色', '青', '青色', '碧', '碧色', '翠', '翠色', '苍', '苍色', '葱', '葱色', '草', '草绿', '嫩', '嫩绿', '深', '深绿', '浅', '浅绿', '淡', '淡绿', '墨', '墨绿', '碧', '碧绿', '翠', '翠绿', '葱', '葱绿', '柳', '柳绿', '松', '松绿', '豆', '豆绿', '茶', '茶绿', '军', '军绿'],
            'yellow': ['黄', 'yellow', '黄色', '金', '金色', '橙', '橙色', '橘', '橘色', '杏', '杏色', '柠', '柠檬', '米', '米色', '驼', '驼色', '土', '土黄', '鹅', '鹅黄', '杏', '杏黄', '橙', '橙黄', '金', '金黄', '橘', '橘黄', '棕', '棕黄', '赭', '赭黄'],
            'white': ['白', 'white', '白色', '雪', '雪白', '乳', '乳白', '奶', '奶白', '米', '米白', '银', '银白', '花', '花白', '灰', '灰白', '素', '素白', '皓', '皓白', '皎', '皎白', '皑', '皑白'],
            'black': ['黑', 'black', '黑色', '墨', '墨色', '乌', '乌色', '玄', '玄色', '黝', '黝色', '黯', '黯色', '皂', '皂色', '漆', '漆黑', '煤', '煤黑', '炭', '炭黑', '铁', '铁黑', '油', '油黑'],
            'gray': ['灰', 'gray', '灰色', '银', '银色', '烟', '烟灰', '炭', '炭灰', '铁', '铁灰', '铅', '铅灰', '苍', '苍白', '暗', '暗灰', '浅', '浅灰', '深', '深灰'],
            'purple': ['紫', 'purple', '紫色', '紫红', '紫红', '紫罗兰', '紫罗兰', '茄', '茄色', '葡', '葡色', '藕', '藕色', '绛', '绛紫', '酱', '酱紫', '黛', '黛紫', '青', '青紫', '蓝', '蓝紫'],
            'orange': ['橙', 'orange', '橙色', '橘', '橘色', '桔', '桔色', '柿', '柿色', '柑', '柑色', '柚', '柚色', '柠', '柠色', '杏', '杏色', '肉', '肉色', '鲑', '鲑色', '虾', '虾色', '蟹', '蟹色', '琥', '琥珀', '珊瑚', '珊瑚'],
            'pink': ['粉', 'pink', '粉色', '粉红', '粉红', '桃', '桃色', '樱', '樱色', '玫', '玫瑰', '玫红', '玫红', '蔷', '蔷薇', '蔷红', '蔷红', '藕', '藕色', '藕荷', '藕荷', '水', '水红', '妃', '妃色', '嫣', '嫣红', '胭脂', '胭脂'],
            'brown': ['棕', 'brown', '棕色', '褐', '褐色', '咖', '咖啡', '栗', '栗色', '茶', '茶色', '驼', '驼色', '赭', '赭色', '土', '土色', '木', '木色', '皮', '皮色', '铜', '铜色', '铁', '铁色', '赭', '赭石', '熟', '熟褐', '生', '生褐'],
        }
        
        # 对象映射（扩展类别）
        self.object_map = {
            # 建筑物
            'building': ['大楼', '建筑', 'building', '建筑物', '高楼', '房子', '房屋', '住宅', '楼房', '大厦', '楼宇', '场馆', '场馆', '体育馆', '体育馆', '展览馆', '展览馆', '博物馆', '博物馆', '图书馆', '图书馆', '剧院', '剧院', '影院', '影院', '商场', '商场', '超市', '超市', '商店', '商店', '店铺', '店铺', '门面', '门面', '门面房', '门面房'],
            # 停车场
            'parking': ['停车', '车位', 'parking', '停车场', '停车位', '停车区', '停车区', '停车库', '停车库', '停车楼', '停车楼', '车场', '车场', '车库', '车库', '车位', '车位', '泊位', '泊位', '泊车位', '泊车位'],
            # 标志
            'sign': ['标志', '标识', 'sign', '指示牌', '标牌', '交通标志', '路标', '路标', '路牌', '路牌', '招牌', '招牌', '广告牌', '广告牌', '标识牌', '标识牌', '警示牌', '警示牌', '提示牌', '提示牌', '引导牌', '引导牌', '方向牌', '方向牌', '名称牌', '名称牌', '门牌', '门牌', '号牌', '号牌', '编号牌', '编号牌'],
            # 灯
            'light': ['灯', '路灯', 'light', '交通灯', '红绿灯', '信号灯', '照明灯', '照明灯', '景观灯', '景观灯', '装饰灯', '装饰灯', '霓虹灯', '霓虹灯', 'LED灯', 'LED灯', '节能灯', '节能灯', '日光灯', '日光灯', '白炽灯', '白炽灯'],
            # 树
            'tree': ['树', '树木', 'tree', '大树', '树林', '林木', '乔木', '乔木', '灌木', '灌木', '松树', '松树', '柏树', '柏树', '杨树', '杨树', '柳树', '柳树', '槐树', '槐树', '榆树', '榆树', '梧桐', '梧桐', '银杏', '银杏', '枫树', '枫树', '桦树', '桦树', '榕树', '榕树', '樟树', '樟树', '桂树', '桂树', '桃树', '桃树', '梨树', '梨树', '杏树', '杏树', '枣树', '枣树', '柿树', '柿树', '果树', '果树'],
            # 车
            'car': ['车', '汽车', 'car', '车辆', '机动车', '轿车', '轿车', '卡车', '卡车', '货车', '货车', '客车', '客车', '面包车', '面包车', '吉普车', '吉普车', '越野车', '越野车', '跑车', '跑车', '电动车', '电动车', '新能源车', '新能源车', '混动车', '混动车', '出租车', '出租车', '网约车', '网约车', '私家车', '私家车', '公务车', '公务车', '警车', '警车', '救护车', '救护车', '消防车', '消防车'],
            # 柱子
            'pole': ['柱子', '灯柱', 'pole', '电线杆', '杆子', '立柱', '立柱', '支柱', '支柱', '支撑柱', '支撑柱', '装饰柱', '装饰柱', '旗杆', '旗杆', '标杆', '标杆', '路桩', '路桩', '界桩', '界桩', '里程碑', '里程碑', '百米桩', '百米桩'],
            # 桥
            'bridge': ['桥', '桥梁', 'bridge', '天桥', '立交桥', '立交桥', '高架桥', '高架桥', '斜拉桥', '斜拉桥', '悬索桥', '悬索桥', '拱桥', '拱桥', '梁桥', '梁桥', '栈桥', '栈桥', '浮桥', '浮桥', '吊桥', '吊桥', '廊桥', '廊桥', '石桥', '石桥', '木桥', '木桥', '铁桥', '铁桥', '钢桥', '钢桥', '混凝土桥', '混凝土桥'],
            # 围栏
            'fence': ['围墙', '栅栏', 'fence', '栏杆', '护栏', '护栏', '围栏', '围栏', '篱笆', '篱笆', '铁丝网', '铁丝网', '防护网', '防护网', '隔离带', '隔离带', '隔离栏', '隔离栏', '防撞栏', '防撞栏', '波形护栏', '波形护栏'],
            # 墙
            'wall': ['墙', '墙壁', 'wall', '墙体', '墙面', '墙面', '围墙', '围墙', '隔墙', '隔墙', '挡土墙', '挡土墙', '护坡墙', '护坡墙', '防火墙', '防火墙', '承重墙', '承重墙', '剪力墙', '剪力墙', '填充墙', '填充墙', '幕墙', '幕墙', '玻璃幕墙', '玻璃幕墙'],
            # 道路
            'road': ['道路', '马路', 'road', '公路', '街道', '街道', '大路', '大路', '小路', '小路', '干道', '干道', '支路', '支路', '辅路', '辅路', '环路', '环路', '快速路', '快速路', '高速路', '高速路', '国道', '国道', '省道', '省道', '县道', '县道', '乡道', '乡道', '村道', '村道', '巷道', '巷道', '胡同', '胡同'],
            # 人行道
            'sidewalk': ['人行道', '步道', 'sidewalk', '便道', '步行道', '步行道', '人行路', '人行路', '散步道', '散步道', '游步道', '游步道', '健身步道', '健身步道', '盲道', '盲道', '缘石坡道', '缘石坡道'],
            # 入口
            'entrance': ['入口', '门口', 'entrance', '大门', '进口', '进口', '进入口', '进入口', '出入口', '出入口', '通道口', '通道口', '门口', '门口', '门洞', '门洞', '门廊', '门廊', '门厅', '门厅', '玄关', '玄关', '前厅', '前厅'],
            # 角落
            'corner': ['角落', '拐角', 'corner', '墙角', '转角', '转角', '路口', '路口', '叉口', '叉口', '交汇处', '交汇处', '交叉口', '交叉口', '十字路口', '十字路口', '丁字路口', '丁字路口', '三岔口', '三岔口', '弯道', '弯道'],
            # 路口
            'junction': ['路口', '交叉口', 'junction', '交汇处', '交叉点', '交叉点', '汇合处', '汇合处', '分流处', '分流处', '转弯处', '转弯处', '掉头处', '掉头处', '待转区', '待转区'],
            # 车库
            'garage': ['车库', '停车库', 'garage', '地下车库', '地下车库', '地上车库', '地上车库', '立体车库', '立体车库', '机械车库', '机械车库', '智能车库', '智能车库'],
            # 箱子
            'box': ['箱子', '盒子', 'box', '方块', '集装箱', '集装箱', '货柜', '货柜', '包装箱', '包装箱', '收纳箱', '收纳箱', '储物箱', '储物箱', '工具箱', '工具箱', '配电箱', '配电箱', '控制箱', '控制箱', '电表箱', '电表箱', '水表箱', '水表箱', '燃气箱', '燃气箱', '信箱', '信箱', '快递柜', '快递柜', '外卖柜', '外卖柜'],
            # 草坪
            'grass': ['草坪', '草地', 'grass', '草皮', '草皮', '绿地', '绿地', '绿化带', '绿化带', '草坪区', '草坪区', '草地广场', '草地广场'],
            # 花坛
            'flowerbed': ['花坛', '花圃', 'flowerbed', '花境', '花境', '花带', '花带', '花池', '花池', '花箱', '花箱', '花架', '花架', '花廊', '花廊', '花园', '花园', '花境', '花境'],
            # 喷泉
            'fountain': ['喷泉', '水景', 'fountain', '喷水池', '喷水池', '水幕墙', '水幕墙', '跌水', '跌水', '涌泉', '涌泉', '旱喷', '旱喷', '音乐喷泉', '音乐喷泉'],
            # 雕塑
            'sculpture': ['雕塑', '雕像', 'sculpture', '塑像', '塑像', '雕刻', '雕刻', '造型', '造型', '艺术装置', '艺术装置', '景观小品', '景观小品'],
            # 座椅
            'bench': ['座椅', '椅子', 'bench', '长椅', '长椅', '长凳', '长凳', '坐凳', '坐凳', '休息椅', '休息椅', '休闲椅', '休闲椅', '公园椅', '公园椅', '景观座椅', '景观座椅'],
            # 桌子
            'table': ['桌子', '桌', 'table', '餐桌', '餐桌', '茶几', '茶几', '方桌', '方桌', '圆桌', '圆桌', '长桌', '长桌', '办公桌', '办公桌', '会议桌', '会议桌', '休闲桌', '休闲桌', '户外桌', '户外桌'],
            # 垃圾桶
            'trash_can': ['垃圾桶', '垃圾箱', 'trash_can', '果皮箱', '果皮箱', '废物箱', '废物箱', '垃圾筒', '垃圾筒', '分类垃圾桶', '分类垃圾桶', '智能垃圾桶', '智能垃圾桶'],
            # 自行车
            'bicycle': ['自行车', '单车', 'bicycle', '脚踏车', '脚踏车', '山地车', '山地车', '公路车', '公路车', '折叠车', '折叠车', '电动自行车', '电动自行车', '共享单车', '共享单车', '公共自行车', '公共自行车'],
            # 摩托车
            'motorcycle': ['摩托车', '摩托', 'motorcycle', '机车', '机车', '电动车', '电动车', '电摩', '电摩', '轻摩', '轻摩', '踏板车', '踏板车', '跨骑车', '跨骑车', '弯梁车', '弯梁车'],
            # 公交车
            'bus': ['公交车', '巴士', 'bus', '公共汽车', '公共汽车', '大巴', '大巴', '中巴', '中巴', '小巴', '小巴', ' BRT', ' BRT', '快速公交', '快速公交', '无轨电车', '无轨电车', '有轨电车', '有轨电车'],
            # 地铁
            'subway': ['地铁', '轨道交通', 'subway', '城铁', '城铁', '轻轨', '轻轨', '有轨', '有轨', '磁悬浮', '磁悬浮', '地铁站', '地铁站', '地铁口', '地铁口', '地铁入口', '地铁入口', '地铁出口', '地铁出口'],
            # 火车
            'train': ['火车', '列车', 'train', '高铁', '高铁', '动车', '动车', '城际', '城际', '普快', '普快', '特快', '特快', '直达', '直达', '货运列车', '货运列车', '客运列车', '客运列车'],
            # 飞机
            'airplane': ['飞机', '航班', 'airplane', '客机', '客机', '货机', '货机', '直升机', '直升机', '无人机', '无人机', '航模', '航模'],
            # 船
            'boat': ['船', '船舶', 'boat', '轮船', '轮船', '货船', '货船', '客船', '客船', '游艇', '游艇', '快艇', '快艇', '帆船', '帆船', '渔船', '渔船', '渡船', '渡船', '驳船', '驳船'],
        }
        
        # 关系映射
        self.relation_map = {
            'near': ['靠近', '邻近', '附近', '旁边', '近', 'beside', 'next to', 'close to', 'adjacent to', 'by', 'around'],
            'between': ['之间', '中间', '当中', 'between', 'among', 'in between', 'in the middle of'],
            'above': ['上方', '上面', '顶部', 'above', 'over', 'on top of', 'at the top of'],
            'below': ['下方', '下面', '底部', 'below', 'under', 'beneath', 'underneath', 'at the bottom of'],
            'in_front_of': ['前面', '前方', '正前方', 'in front of', 'ahead of', 'before'],
            'behind': ['后面', '后方', '背后', 'behind', 'at the back of', 'in the rear of'],
            'left_of': ['左边', '左侧', 'left of', 'to the left of', 'on the left side of'],
            'right_of': ['右边', '右侧', 'right of', 'to the right of', 'on the right side of'],
            'inside': ['里面', '内部', 'inside', 'within', 'in', 'into', 'interior'],
            'outside': ['外面', '外部', 'outside', 'exterior', 'out of', 'beyond'],
            'across_from': ['对面', '对过', 'across from', 'opposite to', 'facing'],
            'along': ['沿着', '顺着', 'along', 'alongside', 'parallel to'],
            'at': ['在', 'at', 'in', 'on', 'located at', 'situated at'],
        }
    
    def parse(self, query: str) -> NLUResult:
        """
        解析查询
        
        Args:
            query: 自然语言查询
            
        Returns:
            NLU解析结果
        """
        start_time = time.time()
        
        # 预处理
        query = self._preprocess(query)
        
        # 提取各个实体
        direction = self._extract_direction(query)
        color = self._extract_color(query)
        obj = self._extract_object(query)
        relation = self._extract_relation(query)
        distance = self._extract_distance(query)
        landmarks = self._extract_landmarks(query)
        intent = self._determine_intent(query)
        
        # 计算置信度
        confidence = self._calculate_confidence(
            direction, color, obj, relation, distance, query
        )
        
        parse_time = (time.time() - start_time) * 1000
        
        return NLUResult(
            direction=direction,
            color=color,
            object=obj,
            relation=relation,
            distance=distance,
            landmarks=landmarks,
            intent=intent,
            confidence=confidence,
            parse_time=parse_time,
            method="advanced_rule"
        )
    
    def _preprocess(self, query: str) -> str:
        """预处理查询"""
        # 去除多余空格
        query = ' '.join(query.split())
        # 转换为小写（英文部分）
        query = query.lower()
        return query
    
    def _extract_direction(self, query: str) -> Optional[str]:
        """提取方向"""
        for direction, keywords in self.direction_map.items():
            for keyword in keywords:
                if keyword in query:
                    return direction
        return None
    
    def _extract_color(self, query: str) -> Optional[str]:
        """提取颜色"""
        for color, keywords in self.color_map.items():
            for keyword in keywords:
                if keyword in query:
                    return color
        return None
    
    def _extract_object(self, query: str) -> Optional[str]:
        """提取对象"""
        # 按关键词长度排序，优先匹配更具体的
        for obj, keywords in sorted(self.object_map.items(), 
                                     key=lambda x: -max(len(k) for k in x[1])):
            for keyword in keywords:
                if keyword in query:
                    return obj
        return None
    
    def _extract_relation(self, query: str) -> Optional[str]:
        """提取空间关系"""
        for relation, keywords in self.relation_map.items():
            for keyword in keywords:
                if keyword in query:
                    return relation
        return None
    
    def _extract_distance(self, query: str) -> Optional[str]:
        """提取距离"""
        match = self.patterns['distance'].search(query)
        if match:
            return f"{match.group(1)}米"
        return None
    
    def _extract_landmarks(self, query: str) -> List[str]:
        """提取地标/参考物"""
        landmarks = []
        
        # 使用"在...附近/旁边"模式
        match = self.patterns['near_pattern'].search(query)
        if match:
            landmark = match.group(1).strip()
            if landmark:
                landmarks.append(landmark)
        
        # 使用"在...的"模式
        pattern = re.compile(r'在\s*(.+?)\s*的')
        matches = pattern.findall(query)
        landmarks.extend([m.strip() for m in matches if m.strip()])
        
        return landmarks
    
    def _determine_intent(self, query: str) -> str:
        """确定意图"""
        # 问候语
        greetings = ['你好', '您好', 'hello', 'hi', 'hey', '早上好', '下午好', '晚上好']
        if any(g in query for g in greetings):
            return "greeting"
        
        # 导航意图
        navigation = ['去', '到', '走', '导航', '路线', '怎么走', '怎么去', '怎么去']
        if any(n in query for n in navigation):
            return "navigation"
        
        # 信息查询
        information = ['是什么', '在哪里', '有什么', '介绍', '说明', '信息']
        if any(i in query for i in information):
            return "information"
        
        # 默认位置查询
        return "location_query"
    
    def _calculate_confidence(self, direction, color, obj, relation, distance, query: str) -> float:
        """计算置信度"""
        scores = []
        
        if direction:
            scores.append(0.85)
        if color:
            scores.append(0.80)
        if obj:
            scores.append(0.85)
        if relation:
            scores.append(0.75)
        if distance:
            scores.append(0.90)
        
        if not scores:
            return 0.20
        
        base_confidence = sum(scores) / len(scores)
        
        # 根据匹配数量调整
        if len(scores) >= 4:
            multiplier = 1.10
        elif len(scores) == 3:
            multiplier = 1.05
        elif len(scores) == 2:
            multiplier = 0.95
        else:
            multiplier = 0.80
        
        # 根据查询长度调整
        query_len = len(query)
        if query_len < 5:
            multiplier *= 0.75
        elif query_len > 100:
            multiplier *= 0.85
        
        return min(base_confidence * multiplier, 0.98)


# 全局解析器实例
_advanced_nlu_parser = None

def get_advanced_nlu_parser() -> AdvancedNLUParser:
    """获取高级NLU解析器实例"""
    global _advanced_nlu_parser
    if _advanced_nlu_parser is None:
        _advanced_nlu_parser = AdvancedNLUParser()
    return _advanced_nlu_parser


if __name__ == "__main__":
    # 测试
    parser = get_advanced_nlu_parser()
    
    test_queries = [
        "找到红色的汽车",
        "在建筑物左侧的树",
        "距离入口10米的地方",
        "蓝色的椅子在停车场",
        "白色的建筑物前面",
    ]
    
    print("高级NLU解析器测试")
    print("=" * 60)
    
    for query in test_queries:
        result = parser.parse(query)
        print(f"\n查询: {query}")
        print(f"  方向: {result.direction}")
        print(f"  颜色: {result.color}")
        print(f"  对象: {result.object}")
        print(f"  关系: {result.relation}")
        print(f"  距离: {result.distance}")
        print(f"  置信度: {result.confidence:.2f}")
        print(f"  耗时: {result.parse_time:.2f}ms")
