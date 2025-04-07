// 游戏常量
const DEBUGGING = false; // 是否开启调试模式
const PLAYER_COUNT = 6; // 玩家数量
const SMALL_BLIND = 10; // 小盲注
const BIG_BLIND = 20; // 大盲注
const STARTING_CHIPS = 1000; // 初始筹码
const DEFAULT_ACTION_TIME = 30; // 默认行动时间(秒)
const MAX_ROUNDS = 20; // 最大回合数
const AUTO_DEAL_DELAY = 1200; // 自动发牌延迟(毫秒)
const HAND_RANKINGS = {
    highCard: 1,
    pair: 2,
    twoPair: 3,
    threeOfAKind: 4,
    straight: 5,
    flush: 6,
    fullHouse: 7,
    fourOfAKind: 8,
    straightFlush: 9,
    royalFlush: 10
};

// 游戏阶段对应的中文描述
const PHASE_DESCRIPTIONS = {
    'waiting': '等待开始',
    'preflop': '前翻牌圈',
    'flop': '翻牌圈',
    'turn': '转牌圈',
    'river': '河牌圈',
    'showdown': '摊牌阶段',
    'finished': '游戏结束'
};

// 动作类型对应的中文描述
const ACTION_DESCRIPTIONS = {
    'fold': '弃牌',
    'check': '让牌',
    'call': '跟注',
    'raise': '加注',
    'allIn': '全押',
    'smallBlind': '小盲',
    'bigBlind': '大盲'
};

// 玩家言语集合
const PLAYER_DIALOGS = {
    fold: {
        'Daniel Negreanu': [
        '不行，这牌打不了。',
        '我弃了，这牌没法打。',
        '这牌太差了，我弃牌。',
        '算了吧，没胜算。',
        '保留实力，下把再战！'
    ],
        'Phil Ivey': [
            '这手牌不行，我弃了。',
            '这牌太差了，我弃牌。',
            '我不想继续下注，我弃牌。',
            '这牌不值得继续下注，我弃牌。',
            '我弃牌，下一把再来。'
        ],
        'Doyle Brunson': [
            '这牌不行，我弃了。',
            '这牌太差了，我弃牌。',
            '我不想继续下注，我弃牌。',
            '这牌不值得继续下注，我弃牌。',
            '我弃牌，下一把再来。'
        ],
        'Phil Hellmuth': [
            '这手牌不行，我弃了。',
            '这牌太差了，我弃牌。',
            '我不想继续下注，我弃牌。',
            '这牌不值得继续下注，我弃牌。',
            '我弃牌，下一把再来。'
        ],
        'Patrik Antonius': [
            '这手牌不行，我弃了。',
            '这牌太差了，我弃牌。',
            '我不想继续下注，我弃牌。',
            '这牌不值得继续下注，我弃牌。',
            '我弃牌，下一把再来。'
        ],
        'Vanessa Selbst': [
            '这手牌不行，我弃了。',
            '这牌太差了，我弃牌。',
            '我不想继续下注，我弃牌。',
            '这牌不值得继续下注，我弃牌。',
            '我弃牌，下一把再来。'
        ],
        'Liv Boeree': [
            '这手牌不行，我弃了。',
            '这牌太差了，我弃牌。',
            '我不想继续下注，我弃牌。',
            '这牌不值得继续下注，我弃牌。',
            '我弃牌，下一把再来。'
        ],
        'Jennifer Harman': [
            '这手牌不行，我弃了。',
            '这牌太差了，我弃牌。',
            '我不想继续下注，我弃牌。',
            '这牌不值得继续下注，我弃牌。',
            '我弃牌，下一把再来。'
        ],
        'Maria Ho': [
            '这手牌不行，我弃了。',
            '这牌太差了，我弃牌。',
            '我不想继续下注，我弃牌。',
            '这牌不值得继续下注，我弃牌。',
            '我弃牌，下一把再来。'
        ],
        'Kathy Liebert': [
            '这手牌不行，我弃了。',
            '这牌太差了，我弃牌。',
            '我不想继续下注，我弃牌。',
            '这牌不值得继续下注，我弃牌。',
            '我弃牌，下一把再来。'
        ],
        'Johnny Chan': [
            '这手牌不行，我弃了。',
            '这牌太差了，我弃牌。',
            '我不想继续下注，我弃牌。',
            '这牌不值得继续下注，我弃牌。',
            '我弃牌，下一把再来。'
        ],
        'Erik Seidel': [
            '这手牌不行，我弃了。',
            '这牌太差了，我弃牌。',
            '我不想继续下注，我弃牌。',
            '这牌不值得继续下注，我弃牌。',
            '我弃牌，下一把再来。'
        ],
        'Tom Dwan': [
            '这手牌不行，我弃了。',
            '这牌太差了，我弃牌。',
            '我不想继续下注，我弃牌。',
            '这牌不值得继续下注，我弃牌。',
            '我弃牌，下一把再来。'
        ],
        'Fedor Holz': [
            '这手牌不行，我弃了。',
            '这牌太差了，我弃牌。',
            '我不想继续下注，我弃牌。',
            '这牌不值得继续下注，我弃牌。',
            '我弃牌，下一把再来。'
        ],
        'Justin Bonomo': [
            '这手牌不行，我弃了。',
            '这牌太差了，我弃牌。',
            '我不想继续下注，我弃牌。',
            '这牌不值得继续下注，我弃牌。',
            '我弃牌，下一把再来。'
        ],
        'default': [
            '我弃牌了。',
            '这手牌打不了。',
            '我放弃这手牌。',
            '弃牌。',
            '我不跟了。'
        ]
    },
    check: {
        'Daniel Negreanu': [
        '让牌，看看情况。',
        '先观望一下吧。',
        '我让牌。',
        '稳住，先不下注。',
        '且慢，我看看其他人的动作。'
    ],
        'Phil Ivey': [
            '我让牌，看看情况。',
            '我先观望一下。',
            '我让牌。',
            '我稳住，先不下注。',
            '我看看其他人的动作。'
        ],
        'Doyle Brunson': [
            '我让牌，看看情况。',
            '我先观望一下。',
            '我让牌。',
            '我稳住，先不下注。',
            '我看看其他人的动作。'
        ],
        'Phil Hellmuth': [
            '我让牌，看看情况。',
            '我先观望一下。',
            '我让牌。',
            '我稳住，先不下注。',
            '我看看其他人的动作。'
        ],
        'Patrik Antonius': [
            '我让牌，看看情况。',
            '我先观望一下。',
            '我让牌。',
            '我稳住，先不下注。',
            '我看看其他人的动作。'
        ],
        'Vanessa Selbst': [
            '我让牌，看看情况。',
            '我先观望一下。',
            '我让牌。',
            '我稳住，先不下注。',
            '我看看其他人的动作。'
        ],
        'Liv Boeree': [
            '我让牌，看看情况。',
            '我先观望一下。',
            '我让牌。',
            '我稳住，先不下注。',
            '我看看其他人的动作。'
        ],
        'Jennifer Harman': [
            '我让牌，看看情况。',
            '我先观望一下。',
            '我让牌。',
            '我稳住，先不下注。',
            '我看看其他人的动作。'
        ],
        'Maria Ho': [
            '我让牌，看看情况。',
            '我先观望一下。',
            '我让牌。',
            '我稳住，先不下注。',
            '我看看其他人的动作。'
        ],
        'Kathy Liebert': [
            '我让牌，看看情况。',
            '我先观望一下。',
            '我让牌。',
            '我稳住，先不下注。',
            '我看看其他人的动作。'
        ],
        'Johnny Chan': [
            '我让牌，看看情况。',
            '我先观望一下。',
            '我让牌。',
            '我稳住，先不下注。',
            '我看看其他人的动作。'
        ],
        'Erik Seidel': [
            '我让牌，看看情况。',
            '我先观望一下。',
            '我让牌。',
            '我稳住，先不下注。',
            '我看看其他人的动作。'
        ],
        'Tom Dwan': [
            '我让牌，看看情况。',
            '我先观望一下。',
            '我让牌。',
            '我稳住，先不下注。',
            '我看看其他人的动作。'
        ],
        'Fedor Holz': [
            '我让牌，看看情况。',
            '我先观望一下。',
            '我让牌。',
            '我稳住，先不下注。',
            '我看看其他人的动作。'
        ],
        'Justin Bonomo': [
            '我让牌，看看情况。',
            '我先观望一下。',
            '我让牌。',
            '我稳住，先不下注。',
            '我看看其他人的动作。'
        ],
        'default': [
            '我让牌。',
            '看看下一张牌。',
            '不加注，让牌。',
            '继续，我让牌。',
            '让牌。'
        ]
    },
    call: {
        'Daniel Negreanu': [
        '跟注！',
        '我跟了！',
        '这把我想看看。',
        '有点意思，我跟了。',
        '看来我得跟注了。'
    ],
        'Phil Ivey': [
            '我跟注！',
            '我跟了！',
            '这把我想看看。',
            '有点意思，我跟了。',
            '看来我得跟注了。'
        ],
        'Doyle Brunson': [
            '我跟注！',
            '我跟了！',
            '这把我想看看。',
            '有点意思，我跟了。',
            '看来我得跟注了。'
        ],
        'Phil Hellmuth': [
            '我跟注！',
            '我跟了！',
            '这把我想看看。',
            '有点意思，我跟了。',
            '看来我得跟注了。'
        ],
        'Patrik Antonius': [
            '我跟注！',
            '我跟了！',
            '这把我想看看。',
            '有点意思，我跟了。',
            '看来我得跟注了。'
        ],
        'Vanessa Selbst': [
            '我跟注！',
            '我跟了！',
            '这把我想看看。',
            '有点意思，我跟了。',
            '看来我得跟注了。'
        ],
        'Liv Boeree': [
            '我跟注！',
            '我跟了！',
            '这把我想看看。',
            '有点意思，我跟了。',
            '看来我得跟注了。'
        ],
        'Jennifer Harman': [
            '我跟注！',
            '我跟了！',
            '这把我想看看。',
            '有点意思，我跟了。',
            '看来我得跟注了。'
        ],
        'Maria Ho': [
            '我跟注！',
            '我跟了！',
            '这把我想看看。',
            '有点意思，我跟了。',
            '看来我得跟注了。'
        ],
        'Kathy Liebert': [
            '我跟注！',
            '我跟了！',
            '这把我想看看。',
            '有点意思，我跟了。',
            '看来我得跟注了。'
        ],
        'Johnny Chan': [
            '我跟注！',
            '我跟了！',
            '这把我想看看。',
            '有点意思，我跟了。',
            '看来我得跟注了。'
        ],
        'Erik Seidel': [
            '我跟注！',
            '我跟了！',
            '这把我想看看。',
            '有点意思，我跟了。',
            '看来我得跟注了。'
        ],
        'Tom Dwan': [
            '我跟注！',
            '我跟了！',
            '这把我想看看。',
            '有点意思，我跟了。',
            '看来我得跟注了。'
        ],
        'Fedor Holz': [
            '我跟注！',
            '我跟了！',
            '这把我想看看。',
            '有点意思，我跟了。',
            '看来我得跟注了。'
        ],
        'Justin Bonomo': [
            '我跟注！',
            '我跟了！',
            '这把我想看看。',
            '有点意思，我跟了。',
            '看来我得跟注了。'
        ],
        'default': [
            '我跟注。',
            '我跟。',
            '跟上。',
            '我看你这个注。',
            '跟注。'
        ]
    },
    raise: {
        'Daniel Negreanu': [
        '加注！',
        '再加点！',
        '我看好这手牌。',
        '大家小心了，我加注！',
        '让你们见识一下。'
    ],
        'Phil Ivey': [
            '我加注！',
            '我再加点！',
            '我看好这手牌。',
            '大家小心了，我加注！',
            '让你们见识一下。'
        ],
        'Doyle Brunson': [
            '我加注！',
            '我再加点！',
            '我看好这手牌。',
            '大家小心了，我加注！',
            '让你们见识一下。'
        ],
        'Phil Hellmuth': [
            '我加注！',
            '我再加点！',
            '我看好这手牌。',
            '大家小心了，我加注！',
            '让你们见识一下。'
        ],
        'Patrik Antonius': [
            '我加注！',
            '我再加点！',
            '我看好这手牌。',
            '大家小心了，我加注！',
            '让你们见识一下。'
        ],
        'Vanessa Selbst': [
            '我加注！',
            '我再加点！',
            '我看好这手牌。',
            '大家小心了，我加注！',
            '让你们见识一下。'
        ],
        'Liv Boeree': [
            '我加注！',
            '我再加点！',
            '我看好这手牌。',
            '大家小心了，我加注！',
            '让你们见识一下。'
        ],
        'Jennifer Harman': [
            '我加注！',
            '我再加点！',
            '我看好这手牌。',
            '大家小心了，我加注！',
            '让你们见识一下。'
        ],
        'Maria Ho': [
            '我加注！',
            '我再加点！',
            '我看好这手牌。',
            '大家小心了，我加注！',
            '让你们见识一下。'
        ],
        'Kathy Liebert': [
            '我加注！',
            '我再加点！',
            '我看好这手牌。',
            '大家小心了，我加注！',
            '让你们见识一下。'
        ],
        'Johnny Chan': [
            '我加注！',
            '我再加点！',
            '我看好这手牌。',
            '大家小心了，我加注！',
            '让你们见识一下。'
        ],
        'Erik Seidel': [
            '我加注！',
            '我再加点！',
            '我看好这手牌。',
            '大家小心了，我加注！',
            '让你们见识一下。'
        ],
        'Tom Dwan': [
            '我加注！',
            '我再加点！',
            '我看好这手牌。',
            '大家小心了，我加注！',
            '让你们见识一下。'
        ],
        'Fedor Holz': [
            '我加注！',
            '我再加点！',
            '我看好这手牌。',
            '大家小心了，我加注！',
            '让你们见识一下。'
        ],
        'Justin Bonomo': [
            '我加注！',
            '我再加点！',
            '我看好这手牌。',
            '大家小心了，我加注！',
            '让你们见识一下。'
        ],
        'default': [
            '我加注。',
            '再加点。',
            '加注。',
            '我要加注。',
            '提高赌注。'
        ]
    },
    allIn: {
        'Daniel Negreanu': [
        '全押！',
        '梭哈！',
        '这把我押上所有筹码！',
        '全部押上，接受挑战吗？',
        '豁出去了！'
    ],
        'Phil Ivey': [
            '我全押！',
            '我梭哈！',
            '这把我押上所有筹码！',
            '全部押上，接受挑战吗？',
            '我豁出去了！'
        ],
        'Doyle Brunson': [
            '我全押！',
            '我梭哈！',
            '这把我押上所有筹码！',
            '全部押上，接受挑战吗？',
            '我豁出去了！'
        ],
        'Phil Hellmuth': [
            '我全押！',
            '我梭哈！',
            '这把我押上所有筹码！',
            '全部押上，接受挑战吗？',
            '我豁出去了！'
        ],
        'Patrik Antonius': [
            '我全押！',
            '我梭哈！',
            '这把我押上所有筹码！',
            '全部押上，接受挑战吗？',
            '我豁出去了！'
        ],
        'Vanessa Selbst': [
            '我全押！',
            '我梭哈！',
            '这把我押上所有筹码！',
            '全部押上，接受挑战吗？',
            '我豁出去了！'
        ],
        'Liv Boeree': [
            '我全押！',
            '我梭哈！',
            '这把我押上所有筹码！',
            '全部押上，接受挑战吗？',
            '我豁出去了！'
        ],
        'Jennifer Harman': [
            '我全押！',
            '我梭哈！',
            '这把我押上所有筹码！',
            '全部押上，接受挑战吗？',
            '我豁出去了！'
        ],
        'Maria Ho': [
            '我全押！',
            '我梭哈！',
            '这把我押上所有筹码！',
            '全部押上，接受挑战吗？',
            '我豁出去了！'
        ],
        'Kathy Liebert': [
            '我全押！',
            '我梭哈！',
            '这把我押上所有筹码！',
            '全部押上，接受挑战吗？',
            '我豁出去了！'
        ],
        'Johnny Chan': [
            '我全押！',
            '我梭哈！',
            '这把我押上所有筹码！',
            '全部押上，接受挑战吗？',
            '我豁出去了！'
        ],
        'Erik Seidel': [
            '我全押！',
            '我梭哈！',
            '这把我押上所有筹码！',
            '全部押上，接受挑战吗？',
            '我豁出去了！'
        ],
        'Tom Dwan': [
            '我全押！',
            '我梭哈！',
            '这把我押上所有筹码！',
            '全部押上，接受挑战吗？',
            '我豁出去了！'
        ],
        'Fedor Holz': [
            '我全押！',
            '我梭哈！',
            '这把我押上所有筹码！',
            '全部押上，接受挑战吗？',
            '我豁出去了！'
        ],
        'Justin Bonomo': [
            '我全押！',
            '我梭哈！',
            '这把我押上所有筹码！',
            '全部押上，接受挑战吗？',
            '我豁出去了！'
        ],
        'default': [
            '全押！',
            '我全押了！',
            '所有筹码都押上！',
            '全部押上！',
            'ALL IN!'
        ]
    },
    win: {
        'Daniel Negreanu': [
        '哈哈，这把是我的了！',
        '谢谢捧场！',
        '我早就知道会赢。',
        '看来今天运气不错！',
        '这就是实力的差距！'
    ],
        'Phil Ivey': [
            '哈哈，这把是我的了！',
            '谢谢捧场！',
            '我早就知道会赢。',
            '看来今天运气不错！',
            '这就是实力的差距！'
        ],
        'Doyle Brunson': [
            '哈哈，这把是我的了！',
            '谢谢捧场！',
            '我早就知道会赢。',
            '看来今天运气不错！',
            '这就是实力的差距！'
        ],
        'Phil Hellmuth': [
            '哈哈，这把是我的了！',
            '谢谢捧场！',
            '我早就知道会赢。',
            '看来今天运气不错！',
            '这就是实力的差距！'
        ],
        'Patrik Antonius': [
            '哈哈，这把是我的了！',
            '谢谢捧场！',
            '我早就知道会赢。',
            '看来今天运气不错！',
            '这就是实力的差距！'
        ],
        'Vanessa Selbst': [
            '哈哈，这把是我的了！',
            '谢谢捧场！',
            '我早就知道会赢。',
            '看来今天运气不错！',
            '这就是实力的差距！'
        ],
        'Liv Boeree': [
            '哈哈，这把是我的了！',
            '谢谢捧场！',
            '我早就知道会赢。',
            '看来今天运气不错！',
            '这就是实力的差距！'
        ],
        'Jennifer Harman': [
            '哈哈，这把是我的了！',
            '谢谢捧场！',
            '我早就知道会赢。',
            '看来今天运气不错！',
            '这就是实力的差距！'
        ],
        'Maria Ho': [
            '哈哈，这把是我的了！',
            '谢谢捧场！',
            '我早就知道会赢。',
            '看来今天运气不错！',
            '这就是实力的差距！'
        ],
        'Kathy Liebert': [
            '哈哈，这把是我的了！',
            '谢谢捧场！',
            '我早就知道会赢。',
            '看来今天运气不错！',
            '这就是实力的差距！'
        ],
        'Johnny Chan': [
            '哈哈，这把是我的了！',
            '谢谢捧场！',
            '我早就知道会赢。',
            '看来今天运气不错！',
            '这就是实力的差距！'
        ],
        'Erik Seidel': [
            '哈哈，这把是我的了！',
            '谢谢捧场！',
            '我早就知道会赢。',
            '看来今天运气不错！',
            '这就是实力的差距！'
        ],
        'Tom Dwan': [
            '哈哈，这把是我的了！',
            '谢谢捧场！',
            '我早就知道会赢。',
            '看来今天运气不错！',
            '这就是实力的差距！'
        ],
        'Fedor Holz': [
            '哈哈，这把是我的了！',
            '谢谢捧场！',
            '我早就知道会赢。',
            '看来今天运气不错！',
            '这就是实力的差距！'
        ],
        'Justin Bonomo': [
            '哈哈，这把是我的了！',
            '谢谢捧场！',
            '我早就知道会赢。',
            '看来今天运气不错！',
            '这就是实力的差距！'
        ]
    },
    thinking: {
        'Daniel Negreanu': [
        '嗯...让我想想...',
        '这牌...',
        '该怎么打呢？',
        '我需要考虑一下...',
        '等等，让我算算...'
        ],
        'Phil Ivey': [
            '嗯...让我想想...',
            '这牌...',
            '该怎么打呢？',
            '我需要考虑一下...',
            '等等，让我算算...'
        ],
        'Doyle Brunson': [
            '嗯...让我想想...',
            '这牌...',
            '该怎么打呢？',
            '我需要考虑一下...',
            '等等，让我算算...'
        ],
        'Phil Hellmuth': [
            '嗯...让我想想...',
            '这牌...',
            '该怎么打呢？',
            '我需要考虑一下...',
            '等等，让我算算...'
        ],
        'Patrik Antonius': [
            '嗯...让我想想...',
            '这牌...',
            '该怎么打呢？',
            '我需要考虑一下...',
            '等等，让我算算...'
        ],
        'Vanessa Selbst': [
            '嗯...让我想想...',
            '这牌...',
            '该怎么打呢？',
            '我需要考虑一下...',
            '等等，让我算算...'
        ],
        'Liv Boeree': [
            '嗯...让我想想...',
            '这牌...',
            '该怎么打呢？',
            '我需要考虑一下...',
            '等等，让我算算...'
        ],
        'Jennifer Harman': [
            '嗯...让我想想...',
            '这牌...',
            '该怎么打呢？',
            '我需要考虑一下...',
            '等等，让我算算...'
        ],
        'Maria Ho': [
            '嗯...让我想想...',
            '这牌...',
            '该怎么打呢？',
            '我需要考虑一下...',
            '等等，让我算算...'
        ],
        'Kathy Liebert': [
            '嗯...让我想想...',
            '这牌...',
            '该怎么打呢？',
            '我需要考虑一下...',
            '等等，让我算算...'
        ],
        'Johnny Chan': [
            '嗯...让我想想...',
            '这牌...',
            '该怎么打呢？',
            '我需要考虑一下...',
            '等等，让我算算...'
        ],
        'Erik Seidel': [
            '嗯...让我想想...',
            '这牌...',
            '该怎么打呢？',
            '我需要考虑一下...',
            '等等，让我算算...'
        ],
        'Tom Dwan': [
            '嗯...让我想想...',
            '这牌...',
            '该怎么打呢？',
            '我需要考虑一下...',
            '等等，让我算算...'
        ],
        'Fedor Holz': [
            '嗯...让我想想...',
            '这牌...',
            '该怎么打呢？',
            '我需要考虑一下...',
            '等等，让我算算...'
        ],
        'Justin Bonomo': [
        '嗯...让我想想...',
        '这牌...',
        '该怎么打呢？',
        '我需要考虑一下...',
        '等等，让我算算...'
    ],
        'default': [
            '让我想想...',
            '嗯...',
            '思考中...',
            '该怎么打呢...',
            '分析局势中...'
        ]
    }
};

// 头像颜色数组
const AVATAR_COLORS = [
    '#3498db', '#e74c3c', '#2ecc71', '#f39c12', 
    '#9b59b6', '#1abc9c', '#d35400', '#c0392b'
];

// 锦标赛盲注级别 - 逐步增加难度的盲注结构
const TOURNAMENT_BLIND_LEVELS = [
    { level: 1, smallBlind: 5, bigBlind: 10, duration: 5 },     // 第1级
    { level: 2, smallBlind: 10, bigBlind: 20, duration: 5 },    // 第2级
    { level: 3, smallBlind: 15, bigBlind: 30, duration: 5 },    // 第3级
    { level: 4, smallBlind: 25, bigBlind: 50, duration: 5 },    // 第4级
    { level: 5, smallBlind: 50, bigBlind: 100, duration: 5 },   // 第5级
    { level: 6, smallBlind: 75, bigBlind: 150, duration: 5 },   // 第6级
    { level: 7, smallBlind: 100, bigBlind: 200, duration: 5 },  // 第7级
    { level: 8, smallBlind: 150, bigBlind: 300, duration: 5 },  // 第8级
    { level: 9, smallBlind: 200, bigBlind: 400, duration: 5 },  // 第9级
    { level: 10, smallBlind: 300, bigBlind: 600, duration: 5 }, // 第10级
    { level: 11, smallBlind: 400, bigBlind: 800, duration: 5 }, // 第11级
    { level: 12, smallBlind: 500, bigBlind: 1000, duration: 5 },// 第12级
    { level: 13, smallBlind: 750, bigBlind: 1500, duration: 5 },// 第13级
    { level: 14, smallBlind: 1000, bigBlind: 2000, duration: 5 },// 第14级
    { level: 15, smallBlind: 1500, bigBlind: 3000, duration: 5 } // 第15级
];

// 锦标赛奖金分配比例 - 百分比表示
const TOURNAMENT_PAYOUTS = [
    { position: 1, percentage: 50 },  // 冠军获得奖池的50%
    { position: 2, percentage: 30 },  // 亚军获得奖池的30%
    { position: 3, percentage: 15 },  // 季军获得奖池的15%
    { position: 4, percentage: 5 }    // 第四名获得奖池的5%
];

// 游戏状态
const gameState = {
    players: [],
    deck: [],
    communityCards: [],
    pot: 0,
    currentPlayer: 0,
    currentBet: 0,
    gamePhase: 'waiting',
    smallBlind: 5,
    bigBlind: 10,
    dealerPosition: 0,
    actionTimer: null,
    timeWarningTimers: [],
    roundNumber: 1,
    soundEnabled: true,
    settings: {
        actionTimeLimit: DEFAULT_ACTION_TIME
    },
    tournament: {
        isEnabled: false,
        startingChips: 1000,
        currentLevel: 0,
        levelDuration: 10, // 默认10分钟一个级别
        levelTimer: null,
        levelTimeRemaining: 0,
        eliminatedPlayers: [],
        isFinished: false,
        prizePool: 10000 // 默认奖池
    }
};

// 扑克牌常量
const SUITS = ['♥', '♦', '♣', '♠'];
const CARD_VALUES = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A'];

// DOM元素
const welcomeScreen = document.getElementById('welcomeScreen');
const gameScreen = document.getElementById('gameScreen');
const playerNameInput = document.getElementById('playerName');
const startGameBtn = document.getElementById('startGameBtn');
const startRoundBtn = document.getElementById('startRoundBtn');
const restartBtn = document.getElementById('restartBtn');
const playerNameDisplay = document.getElementById('playerNameDisplay');
const playerChipsDisplay = document.getElementById('playerChipsAmount');
const potDisplay = document.getElementById('pot');
const potChipsDisplay = document.getElementById('potChips');
const playersContainer = document.getElementById('playersContainer');
const animationContainer = document.getElementById('animationContainer');
const dialogContainer = document.getElementById('dialogContainer');
const gamePhaseDisplay = document.getElementById('gamePhase');
const currentTurnDisplay = document.getElementById('currentTurn');
const dealerButton = document.getElementById('dealerButton');
const communityCardElements = [
    document.getElementById('communityCard1'),
    document.getElementById('communityCard2'),
    document.getElementById('communityCard3'),
    document.getElementById('communityCard4'),
    document.getElementById('communityCard5')
];
const playerCardElements = [
    document.getElementById('playerCard1'),
    document.getElementById('playerCard2')
];
const foldBtn = document.getElementById('foldBtn');
const checkBtn = document.getElementById('checkBtn');
const callBtn = document.getElementById('callBtn');
const raiseBtn = document.getElementById('raiseBtn');
const allInBtn = document.getElementById('allInBtn');
const betSlider = document.getElementById('betSlider');
const betAmount = document.getElementById('betAmount');
const enableSoundCheckbox = document.getElementById('enableSound');
const historyContent = document.getElementById('historyContent');
const roundNumberDisplay = document.getElementById('roundNumber');
const tournamentResults = document.getElementById('tournamentResults');
const tournamentWinnerDisplay = document.getElementById('tournamentWinner');
const resultsTableBody = document.getElementById('resultsTableBody');
const currentLevelDisplay = document.getElementById('currentLevel');
const currentBlindsDisplay = document.getElementById('currentBlinds');
const playersRemainingDisplay = document.getElementById('playersRemaining');
const levelTimeRemainingDisplay = document.getElementById('levelTimeRemaining');
const timerProgressBar = document.getElementById('timerProgressBar');

// 音效元素
const sounds = {
    cardDeal: document.getElementById('cardDealSound'),
    chipStack: document.getElementById('chipStackSound'),
    fold: document.getElementById('foldSound'),
    check: document.getElementById('checkSound'),
    call: document.getElementById('callSound'),
    raise: document.getElementById('raiseSound'),
    win: document.getElementById('winSound'),
    levelUp: document.getElementById('levelUpSound'),
    elimination: document.getElementById('eliminationSound'),
    timeWarning: document.getElementById('timeWarningSound')
};

// 定义知名扑克玩家数据库
const POKER_PLAYERS_DATABASE = [
    {
        id: 1,
        name: "Daniel Negreanu",
        avatarUrl: "https://www.cardplayer.com/assets/players/000/033/714/profile/daniel-negreanu.jpg",
        gender: "male",
        nickname: "Kid Poker",
        country: "Canada",
        achievements: "6 WSOP 金手链, 2 WPT 冠军",
        style: "读牌高手，善于观察对手",
        characteristic: "乐观健谈，总是面带微笑"
    },
    {
        id: 2,
        name: "Phil Ivey",
        avatarUrl: "https://www.pokernews.com/img/incl/photos/phil-ivey-wsop-2019-nolan-dalla.jpg",
        gender: "male",
        nickname: "扑克之虎",
        country: "USA",
        achievements: "10 WSOP 金手链, 1 WPT 冠军",
        style: "极具侵略性，全能型选手",
        characteristic: "扑克脸，冷静沉着"
    },
    {
        id: 3,
        name: "Doyle Brunson",
        avatarUrl: "https://www.pokernews.com/img/incl/photos/doyle-brunson-2.jpg",
        gender: "male",
        nickname: "德州教父",
        country: "USA",
        achievements: "10 WSOP 金手链，德州扑克名人堂",
        style: "老派激进，不惧风险",
        characteristic: "戴着牛仔帽，善于讲故事"
    },
    {
        id: 4,
        name: "Phil Hellmuth",
        avatarUrl: "https://www.pokernews.com/img/incl/photos/phil-hellmuth-1-big.jpg",
        gender: "male",
        nickname: "扑克恶魔",
        country: "USA",
        achievements: "16 WSOP 金手链，最多WSOP冠军得主",
        style: "紧凶型，擅长锦标赛",
        characteristic: "情绪化，容易暴怒，自信心极强"
    },
    {
        id: 5,
        name: "Patrik Antonius",
        avatarUrl: "https://i.pinimg.com/originals/87/e8/a6/87e8a69d89651a56c35c81ade6144e83.jpg",
        gender: "male",
        nickname: "芬兰帅哥",
        country: "Finland",
        achievements: "高额现金局传奇",
        style: "数学型选手，大底池专家",
        characteristic: "英俊外表，冷静沉着"
    },
    {
        id: 6,
        name: "Vanessa Selbst",
        avatarUrl: "https://www.pokernews.com/img/incl/photos/vanessaselbst-big.jpg",
        gender: "female",
        nickname: "女王",
        country: "USA",
        achievements: "3 WSOP 金手链，女子扑克第一人",
        style: "极具侵略性，敢于做大注码",
        characteristic: "自信，直言不讳"
    },
    {
        id: 7,
        name: "Liv Boeree",
        avatarUrl: "https://www.pokernews.com/img/incl/photos/liv-boeree-trophy.jpg",
        gender: "female",
        nickname: "铁淑女",
        country: "UK",
        achievements: "EPT 圣雷莫站冠军，1 WSOP 金手链",
        style: "数学计算型，擅长锦标赛",
        characteristic: "物理学学位，摇滚乐手"
    },
    {
        id: 8,
        name: "Tom Dwan",
        avatarUrl: "https://www.pokernews.com/img/incl/photos/tom-dwan-poker-after-dark.jpg",
        gender: "male",
        nickname: "durrrr",
        country: "USA",
        achievements: "线上传奇，高额现金局大师",
        style: "超级激进，创新型玩法",
        characteristic: "扑克脸，极具攻击性"
    },
    {
        id: 9,
        name: "Fedor Holz",
        avatarUrl: "https://www.pokernews.com/img/incl/photos/fedor-holz-2018-wsop-chidwick.jpg",
        gender: "male",
        nickname: "CrownUpGuy",
        country: "Germany",
        achievements: "1 WSOP 金手链，超高额锦标赛之王",
        style: "锦标赛专家，GTO理论应用",
        characteristic: "年轻有为，心态平和"
    },
    {
        id: 10,
        name: "Jennifer Harman",
        avatarUrl: "https://www.pokernews.com/img/incl/photos/jennifer-harman-2018-wsop.jpg",
        gender: "female",
        nickname: "铁杆珍",
        country: "USA",
        achievements: "2 WSOP 金手链，扑克名人堂",
        style: "高额混合游戏大师",
        characteristic: "坚韧不拔，扑克界女性先驱"
    },
    {
        id: 11,
        name: "Justin Bonomo",
        avatarUrl: "https://www.pokernews.com/img/incl/photos/justin-bonomo-2018-one-drop.jpg",
        gender: "male",
        nickname: "ZeeJustin",
        country: "USA",
        achievements: "3 WSOP 金手链，累计奖金最多的选手",
        style: "GTO理论大师，锦标赛专家",
        characteristic: "理性分析，保持冷静"
    },
    {
        id: 12,
        name: "Maria Ho",
        avatarUrl: "https://www.pokernews.com/img/incl/photos/maria-ho-2019-wsop-main-event.jpg",
        gender: "female",
        nickname: "扑克女战神",
        country: "USA/Taiwan",
        achievements: "多次WSOP决赛桌，扑克评论员",
        style: "全能型，擅长锦标赛",
        characteristic: "优雅专业，扑克解说员"
    },
    {
        id: 13,
        name: "Johnny Chan",
        avatarUrl: "https://www.pokernews.com/img/incl/photos/johnny-chan-wsop.jpg",
        gender: "male",
        nickname: "东方快车",
        country: "USA/China",
        achievements: "10 WSOP 金手链，扑克名人堂",
        style: "善于诱敌深入，陷阱大师",
        characteristic: "手拿幸运橙子，电影《赌王之王》原型"
    },
    {
        id: 14,
        name: "Erik Seidel",
        avatarUrl: "https://www.pokernews.com/img/incl/photos/erik-seidel-1.jpg",
        gender: "male",
        nickname: "沉默刺客",
        country: "USA",
        achievements: "9 WSOP 金手链，扑克名人堂",
        style: "稳健型，锦标赛大师",
        characteristic: "低调内敛，极少交谈"
    },
    {
        id: 15,
        name: "Kathy Liebert",
        avatarUrl: "https://www.pokernews.com/img/incl/photos/kathy-liebert-wsop.jpg",
        gender: "female",
        nickname: "扑克凯蒂",
        country: "USA",
        achievements: "1 WSOP 金手链，多项女子赛事冠军",
        style: "稳健型，锦标赛专家",
        characteristic: "持久战型选手，坚韧不拔"
    },
    {
        id: 16, 
        name: "Phil Laak",
        avatarUrl: "https://www.pokernews.com/img/incl/photos/phil-laak-unabomber.jpg",
        gender: "male",
        nickname: "疯子",
        country: "USA",
        achievements: "1 WSOP 金手链，不眠马拉松纪录保持者",
        style: "不按常理出牌，难以预测",
        characteristic: "佩戴太阳镜和帽衫，行为古怪"
    },
    {
        id: 17,
        name: "Sam Trickett",
        avatarUrl: "https://www.pokernews.com/img/incl/photos/sam-trickett-wsop.jpg",
        gender: "male",
        nickname: "英伦小天王",
        country: "UK",
        achievements: "多项高额锦标赛冠军",
        style: "高额现金局大师，敢于下大注",
        characteristic: "英国最成功的扑克玩家之一"
    },
    {
        id: 18,
        name: "Scotty Nguyen",
        avatarUrl: "https://www.pokernews.com/img/incl/photos/scotty-nguyen-wsop.jpg",
        gender: "male",
        nickname: "王子",
        country: "Vietnam/USA",
        achievements: "5 WSOP 金手链，1998年世界冠军",
        style: "灵活多变，善于心理战",
        characteristic: "标志性笑容，\"Baby\" 口头禅"
    },
    {
        id: 19,
        name: "Antonio Esfandiari",
        avatarUrl: "https://www.pokernews.com/img/incl/photos/antonio-esfandiari-wsop.jpg",
        gender: "male",
        nickname: "魔术师",
        country: "USA",
        achievements: "3 WSOP 金手链，最大买入比赛冠军",
        style: "变幻莫测，擅长读牌",
        characteristic: "前魔术师，社交型玩家"
    },
    {
        id: 20,
        name: "Gus Hansen",
        avatarUrl: "https://www.pokernews.com/img/incl/photos/gus-hansen-fullsize.jpg",
        gender: "male",
        nickname: "伟大的丹麦人",
        country: "Denmark",
        achievements: "3 WPT 冠军，高额现金局传奇",
        style: "超级松凶，非常规打法",
        characteristic: "帅气外表，乐于冒险，全能型运动员"
    },
    {
        id: 21,
        name: "Bryn Kenney",
        avatarUrl: "https://www.pokernews.com/img/incl/photos/bryn-kenney-aussie-millions.jpg",
        gender: "male",
        nickname: "累积奖金王",
        country: "USA",
        achievements: "1 WSOP 金手链，超高额买入比赛专家",
        style: "激进型，锦标赛大师",
        characteristic: "个性张扬，自信满满"
    },
    {
        id: 22,
        name: "Kristen Bicknell",
        avatarUrl: "https://www.pokernews.com/img/incl/photos/kristen-bicknell-partypoker.jpg",
        gender: "female",
        nickname: "Ultimate Grinder",
        country: "Canada",
        achievements: "3 WSOP 金手链，年度最佳女性选手",
        style: "稳健型，扎实基本功",
        characteristic: "勤奋刻苦，以前是线上grinder"
    },
    {
        id: 23,
        name: "Phil Galfond",
        avatarUrl: "https://www.pokernews.com/img/incl/photos/phil-galfond-2017-wsop.jpg",
        gender: "male",
        nickname: "OMGClayAiken",
        country: "USA",
        achievements: "3 WSOP 金手链，线上PLO王者",
        style: "数学型，理论大师",
        characteristic: "扑克哲学家，开设RunItOnce培训网站"
    },
    {
        id: 24,
        name: "Jason Mercier",
        avatarUrl: "https://www.pokernews.com/img/incl/photos/jason-mercier-trophy.jpg",
        gender: "male",
        nickname: "treysfull21",
        country: "USA",
        achievements: "5 WSOP 金手链，前世界第一",
        style: "全能型，擅长混合游戏",
        characteristic: "冷静理性，家庭优先"
    },
    {
        id: 25,
        name: "Vanessa Selbst",
        avatarUrl: "https://www.pokernews.com/img/incl/photos/vanessa-selbst-2013-wsop.jpg",
        gender: "female",
        nickname: "V Money",
        country: "USA",
        achievements: "3 WSOP 金手链，女性扑克奖金第一名",
        style: "激进型，压迫性打法",
        characteristic: "律师背景，敏锐思维，退役后从事投资"
    },
    {
        id: 26,
        name: "Maria Ho",
        avatarUrl: "https://www.pokernews.com/img/incl/photos/maria-ho-2019-wsop.jpg",
        gender: "female",
        nickname: "红桃皇后",
        country: "USA/Taiwan",
        achievements: "WSOP最后女性，多次决赛桌",
        style: "灵活型，锦标赛专家",
        characteristic: "主持人和解说员，聪明且外向"
    },
    {
        id: 27,
        name: "Liv Boeree",
        avatarUrl: "https://www.pokernews.com/img/incl/photos/liv-boeree-2015-wsop.jpg",
        gender: "female",
        nickname: "铁淑女",
        country: "UK",
        achievements: "EPT冠军，WSOP金手链获得者",
        style: "数学型，精准计算",
        characteristic: "物理学学位，乐队吉他手，科普传播者"
    },
    {
        id: 28,
        name: "Jennifer Harman",
        avatarUrl: "https://www.pokernews.com/img/incl/photos/jennifer-harman-wsop.jpg",
        gender: "female",
        nickname: "高额赛场女皇",
        country: "USA",
        achievements: "2 WSOP 金手链，最早打高额现金局的女性",
        style: "紧凶型，读牌大师",
        characteristic: "克服健康问题，扑克名人堂成员"
    },
    {
        id: 29,
        name: "Johnny Chan",
        avatarUrl: "https://www.pokernews.com/img/incl/photos/johnny-chan-wsop-2010.jpg",
        gender: "male",
        nickname: "东方快车",
        country: "China/USA",
        achievements: "10 WSOP 金手链，两届世界冠军",
        style: "精确计算，耐心等待",
        characteristic: "电影《掉包人生》原型，大满贯得主"
    },
    {
        id: 30,
        name: "Elton Tsang",
        avatarUrl: "https://www.pokernews.com/img/incl/photos/elton-tsang-one-drop.jpg",
        gender: "male",
        nickname: "融资高手",
        country: "Hong Kong",
        achievements: "WSOP百万欧元买入冠军",
        style: "灵活型，大底池思路",
        characteristic: "商业投资人，高额买入赛事专家"
    },
    {
        id: 31,
        name: "Winfred Yu",
        avatarUrl: "https://www.pokernews.com/img/incl/photos/winfred-yu-asian-poker-tour.jpg",
        gender: "male",
        nickname: "澳门先生",
        country: "Hong Kong",
        achievements: "亚洲扑克巡回赛多次冠军",
        style: "实战型，现金局高手",
        characteristic: "澳门扑克圈重要人物，扑克王俱乐部创始人"
    },
    {
        id: 32,
        name: "Celina Lin",
        avatarUrl: "https://www.pokernews.com/img/incl/photos/celina-lin-macau.jpg",
        gender: "female",
        nickname: "亚洲皇后",
        country: "China/Australia",
        achievements: "亚洲第一位女性扑克冠军",
        style: "平衡型，锦标赛专家",
        characteristic: "亚洲扑克推广大使，澳门红人"
    },
    {
        id: 33,
        name: "Xuan Liu",
        avatarUrl: "https://www.pokernews.com/img/incl/photos/xuan-liu-poker.jpg",
        gender: "female",
        nickname: "数学精灵",
        country: "China/Canada",
        achievements: "PCA和EPT多次决赛桌",
        style: "计算型，稳健打法",
        characteristic: "学霸背景，多语言天才"
    }
];

// 备用头像集，当真实URL不可用时使用
const FALLBACK_AVATARS = {
    male: [
        "https://randomuser.me/api/portraits/men/32.jpg",
        "https://randomuser.me/api/portraits/men/45.jpg", 
        "https://randomuser.me/api/portraits/men/22.jpg",
        "https://randomuser.me/api/portraits/men/56.jpg",
        "https://randomuser.me/api/portraits/men/18.jpg",
        "https://randomuser.me/api/portraits/men/61.jpg",
        "https://randomuser.me/api/portraits/men/74.jpg",
        "https://randomuser.me/api/portraits/men/39.jpg",
        "https://randomuser.me/api/portraits/men/28.jpg",
        "https://randomuser.me/api/portraits/men/53.jpg",
        "https://randomuser.me/api/portraits/men/68.jpg",
        "https://randomuser.me/api/portraits/men/93.jpg"
    ],
    female: [
        "https://randomuser.me/api/portraits/women/33.jpg",
        "https://randomuser.me/api/portraits/women/21.jpg",
        "https://randomuser.me/api/portraits/women/45.jpg",
        "https://randomuser.me/api/portraits/women/57.jpg",
        "https://randomuser.me/api/portraits/women/69.jpg",
        "https://randomuser.me/api/portraits/women/27.jpg",
        "https://randomuser.me/api/portraits/women/83.jpg",
        "https://randomuser.me/api/portraits/women/17.jpg",
        "https://randomuser.me/api/portraits/women/39.jpg",
        "https://randomuser.me/api/portraits/women/51.jpg",
        "https://randomuser.me/api/portraits/women/62.jpg",
        "https://randomuser.me/api/portraits/women/79.jpg"
    ],
    // 添加更多风格的头像
    cartoon: [
        "https://api.multiavatar.com/1.svg",
        "https://api.multiavatar.com/2.svg",
        "https://api.multiavatar.com/3.svg",
        "https://api.multiavatar.com/4.svg",
        "https://api.multiavatar.com/5.svg",
        "https://api.multiavatar.com/6.svg",
        "https://api.multiavatar.com/7.svg",
        "https://api.multiavatar.com/8.svg"
    ],
    abstract: [
        "https://api.dicebear.com/6.x/identicon/svg?seed=player1",
        "https://api.dicebear.com/6.x/identicon/svg?seed=player2",
        "https://api.dicebear.com/6.x/identicon/svg?seed=player3",
        "https://api.dicebear.com/6.x/identicon/svg?seed=player4",
        "https://api.dicebear.com/6.x/identicon/svg?seed=player5",
        "https://api.dicebear.com/6.x/identicon/svg?seed=player6"
    ]
};

// 玩家位置定义 (相对于窗口的百分比)
const PLAYER_POSITIONS = [
    { top: '80%', left: '50%' },     // 玩家自己的位置（底部中间）
    { top: '60%', left: '20%' },     // 玩家2（左下）
    { top: '30%', left: '10%' },     // 玩家3（左中）
    { top: '10%', left: '30%' },     // 玩家4（左上）
    { top: '10%', left: '70%' },     // 玩家5（右上）
    { top: '30%', left: '90%' },     // 玩家6（右中）
    { top: '60%', left: '80%' }      // 玩家7（右下）
];

// 初始化游戏
function initGame() {
    // 声音设置
    enableSoundCheckbox.addEventListener('change', () => {
        gameState.soundEnabled = enableSoundCheckbox.checked;
    });
    
    // 行动时间限制设置
    const actionTimeLimitSelect = document.getElementById('actionTimeLimit');
    actionTimeLimitSelect.addEventListener('change', () => {
        gameState.settings.actionTimeLimit = parseInt(actionTimeLimitSelect.value);
        console.log(`行动时间限制设置为: ${gameState.settings.actionTimeLimit}秒`);
    });
    
    // 设置默认行动时间限制
    gameState.settings.actionTimeLimit = parseInt(actionTimeLimitSelect.value);

    // 游戏模式选择
    const gameModeOptions = document.querySelectorAll('.game-mode-option');
    const tournamentOptions = document.getElementById('tournamentOptions');
    
    // 监听游戏模式选择
    gameModeOptions.forEach(option => {
        option.addEventListener('click', () => {
            // 移除其他模式的选中状态
            gameModeOptions.forEach(opt => opt.classList.remove('selected'));
            // 添加当前模式的选中状态
            option.classList.add('selected');
            
            const selectedMode = option.getAttribute('data-mode');
            
            // 根据选择显示或隐藏锦标赛选项
            if (selectedMode === 'tournament') {
                tournamentOptions.style.display = 'block';
                gameState.tournament.isEnabled = true;
            } else {
                tournamentOptions.style.display = 'none';
                gameState.tournament.isEnabled = false;
            }
        });
    });
    
    // 锦标赛选项设置
    const startingChipsSelect = document.getElementById('startingChips');
    const levelDurationSelect = document.getElementById('levelDuration');
    
    // 监听锦标赛选项变化
    startingChipsSelect.addEventListener('change', () => {
        gameState.tournament.startingChips = parseInt(startingChipsSelect.value);
    });
    
    levelDurationSelect.addEventListener('change', () => {
        gameState.tournament.levelDuration = parseInt(levelDurationSelect.value);
    });
    
    // 设置默认值
    gameState.tournament.startingChips = parseInt(startingChipsSelect.value);
    gameState.tournament.levelDuration = parseInt(levelDurationSelect.value);

    // 登录逻辑
    startGameBtn.addEventListener('click', () => {
        const playerName = playerNameInput.value.trim();
        if (playerName) {
            startGame(playerName);
        } else {
            alert('请输入玩家名称！');
        }
    });
    
    // 开始牌局按钮事件
    startRoundBtn.addEventListener('click', () => {
        startRoundBtn.classList.add('hidden'); // 隐藏按钮
        addHistoryItem('系统', '牌局开始', null, true);
        startNewRound(); // 开始新一轮
    });
    
    // 默认隐藏开始牌局按钮
    startRoundBtn.classList.add('hidden');

    // 重新开始按钮事件
    restartBtn.addEventListener('click', handleRestart);

    // 游戏控制按钮事件
    foldBtn.addEventListener('click', handleFold);
    checkBtn.addEventListener('click', handleCheck);
    callBtn.addEventListener('click', handleCall);
    raiseBtn.addEventListener('click', handleRaise);
    allInBtn.addEventListener('click', handleAllIn);
    
    // 下注滑块事件
    betSlider.addEventListener('input', updateBetAmount);
    
    // 返回大厅按钮
    if (document.getElementById('backToLobbyBtn')) {
        document.getElementById('backToLobbyBtn').addEventListener('click', () => {
            // 隐藏锦标赛结果和游戏屏幕
            if (document.getElementById('tournamentResults')) {
                document.getElementById('tournamentResults').style.display = 'none';
            }
            
            gameScreen.style.display = 'none';
            welcomeScreen.style.display = 'block';
            
            // 重置游戏状态
            resetGameState();
        });
    }

    // 初始化荷官按钮位置
    updateDealerButton(0);
}

// 重置游戏状态
function resetGameState() {
    // 清除计时器
    clearActionTimer();
    
    // 如果有锦标赛计时器，也清除
    if (gameState.tournament.levelTimer) {
        clearInterval(gameState.tournament.levelTimer);
        gameState.tournament.levelTimer = null;
    }
    
    // 重置基本游戏状态
    gameState.players = [];
    gameState.deck = [];
    gameState.communityCards = [];
    gameState.pot = 0;
    gameState.currentPlayer = 0;
    gameState.currentBet = 0;
    gameState.gamePhase = 'waiting';
    gameState.dealerPosition = 0;
    gameState.roundNumber = 1;
    
    // 重置锦标赛状态
    gameState.tournament.currentLevel = 0;
    gameState.tournament.levelTimeRemaining = 0;
    gameState.tournament.eliminatedPlayers = [];
    gameState.tournament.isFinished = false;
    
    // 清空历史记录
    clearHistory();
    
    // 清除UI元素
    playersContainer.innerHTML = '';
    dialogContainer.innerHTML = '';
    animationContainer.innerHTML = '';
    
    // 清空卡牌显示
    playerCardElements.forEach(element => {
        element.textContent = '';
        element.className = 'card-placeholder';
        element.style.backgroundColor = 'rgba(255, 255, 255, 0.1)';
    });
    
    communityCardElements.forEach(element => {
        element.textContent = '';
        element.className = 'card-placeholder';
        element.style.backgroundColor = 'rgba(255, 255, 255, 0.1)';
    });
    
    // 重置显示
    roundNumberDisplay.textContent = 1;
    potDisplay.textContent = '底池: $0';
    potChipsDisplay.innerHTML = '';
    gamePhaseDisplay.textContent = PHASE_DESCRIPTIONS.waiting;
    currentTurnDisplay.textContent = '轮到: -';
}

// 播放声音
function playSound(soundName) {
    if (gameState.soundEnabled && sounds[soundName]) {
        // 重置播放位置
        sounds[soundName].currentTime = 0;
        sounds[soundName].play().catch(error => {
            console.log('播放声音失败:', error);
        });
    }
}

// 创建动画通知
function createActionNotification(text, position) {
    const notification = document.createElement('div');
    notification.className = 'action-notification';
    notification.textContent = text;
    notification.style.left = position.left;
    notification.style.top = position.top;
    
    animationContainer.appendChild(notification);
    
    // 动画结束后删除元素
    notification.addEventListener('animationend', () => {
        notification.remove();
    });
}

// 创建对话气泡
function createDialogBubble(text, position, playerId) {
    // 检查是否已有该玩家的对话气泡，如果有则移除
    const existingDialog = document.querySelector(`.dialog-bubble[data-player-id="${playerId}"]`);
    if (existingDialog) {
        existingDialog.remove();
    }
    
    const dialog = document.createElement('div');
    dialog.className = 'dialog-bubble';
    dialog.textContent = text;
    dialog.setAttribute('data-player-id', playerId);
    
    // 设置位置
    dialog.style.left = `calc(${position.left} - 75px)`;
    dialog.style.top = `calc(${position.top} - 80px)`;
    
    dialogContainer.appendChild(dialog);
    
    // 动画结束后删除元素
    dialog.addEventListener('animationend', () => {
        dialog.remove();
    });
}

// 获取随机对话
function getRandomDialog(type, playerName) {
    // 确保对话类型存在
    if (!PLAYER_DIALOGS[type]) {
        console.log(`对话类型 ${type} 不存在，使用默认对话`);
        return '...';
    }
    
    // 如果有直接的数组（非嵌套结构）
    if (Array.isArray(PLAYER_DIALOGS[type])) {
        const dialogArray = PLAYER_DIALOGS[type];
        return dialogArray[Math.floor(Math.random() * dialogArray.length)];
    }
    
    // 尝试获取特定玩家的对话
    let dialogs;
    if (playerName && PLAYER_DIALOGS[type][playerName]) {
        dialogs = PLAYER_DIALOGS[type][playerName];
    } else if (PLAYER_DIALOGS[type]['default']) {
        // 如果没有找到特定玩家的对话，使用默认对话
        console.log(`玩家 ${playerName} 的 ${type} 对话不存在，使用默认对话`);
        dialogs = PLAYER_DIALOGS[type]['default'];
    } else {
        // 如果没有默认对话，返回一个固定的值
        console.log(`没有 ${type} 类型的默认对话，返回固定值`);
        return '...';
    }
    
    // 随机选择一个对话
    return dialogs[Math.floor(Math.random() * dialogs.length)];
}

// 生成玩家头像颜色
function generateAvatarColor(index) {
    return AVATAR_COLORS[index % AVATAR_COLORS.length];
}

// 创建筹码动画
function createChipAnimation(startPos, endPos, amount) {
    const numChips = Math.min(5, Math.max(1, Math.floor(amount / 10)));
    
    for (let i = 0; i < numChips; i++) {
        setTimeout(() => {
            const chip = document.createElement('div');
            chip.className = 'chip-animation';
            
            // 随机起始位置偏移
            const randomOffsetX = Math.random() * 20 - 10;
            const randomOffsetY = Math.random() * 20 - 10;
            chip.style.left = `calc(${startPos.left} + ${randomOffsetX}px)`;
            chip.style.top = `calc(${startPos.top} + ${randomOffsetY}px)`;
            
            // 计算终点位置
            const endX = `calc(${endPos.left} - ${chip.style.left})`;
            const endY = `calc(${endPos.top} - ${chip.style.top})`;
            
            chip.style.setProperty('--end-x', endX);
            chip.style.setProperty('--end-y', endY);
            
            // 添加动画
            animationContainer.appendChild(chip);
            
            // 添加动画
            chip.style.animation = `moveChip ${0.5 + Math.random() * 0.5}s ease-out forwards`;
            
            // 动画结束后删除元素
            chip.addEventListener('animationend', () => {
                chip.remove();
                if (i === numChips - 1) {
                    playSound('chipStack');
                }
            });
        }, i * 100);
    }
}

// 更新筹码显示
function updateChipsDisplay() {
    const mainPlayer = gameState.players[0];
    playerChipsDisplay.textContent = mainPlayer.chips;
    
    // 更新底池筹码图标
    updatePotChips();
}

// 更新底池筹码图标
function updatePotChips() {
    potChipsDisplay.innerHTML = '';
    const numChips = Math.min(10, Math.max(1, Math.floor(gameState.pot / 50)));
    
    for (let i = 0; i < numChips; i++) {
        const chip = document.createElement('div');
        chip.className = 'pot-chip';
        // 随机旋转和位置
        const randomRotate = Math.random() * 30 - 15;
        chip.style.transform = `rotate(${randomRotate}deg)`;
        chip.style.marginTop = `${Math.random() * 4 - 2}px`;
        potChipsDisplay.appendChild(chip);
    }
}

// 更新游戏阶段显示
function updateGamePhaseDisplay() {
    gamePhaseDisplay.textContent = PHASE_DESCRIPTIONS[gameState.gamePhase] || '游戏进行中';
}

// 更新当前玩家显示
function updateCurrentPlayerDisplay() {
    if (gameState.currentPlayer >= 0 && gameState.currentPlayer < gameState.players.length) {
        const currentPlayer = gameState.players[gameState.currentPlayer];
        currentTurnDisplay.textContent = `轮到: ${currentPlayer.name}`;
        
        // 高亮当前玩家
        document.querySelectorAll('.player-position').forEach(el => {
            el.classList.remove('current-player');
        });
        
        const playerElement = document.getElementById(`player${gameState.currentPlayer}`);
        if (playerElement) {
            playerElement.classList.add('current-player');
        }
    } else {
        currentTurnDisplay.textContent = '轮到: -';
    }
}

// 更新荷官按钮位置
function updateDealerButton(position) {
    if (position >= 0 && position < PLAYER_POSITIONS.length) {
        dealerButton.style.left = `calc(${PLAYER_POSITIONS[position].left} - 15px)`;
        dealerButton.style.top = `calc(${PLAYER_POSITIONS[position].top} - 15px)`;
    } else {
        // 默认隐藏
        dealerButton.style.display = 'none';
    }
}

// 添加吴彦祖头像URL和更多真实头像
const MAIN_PLAYER_AVATAR = "https://m.media-amazon.com/images/M/MV5BMjA2NTI1ODk3N15BMl5BanBnXkFtZTcwODUyMTM1MQ@@._V1_.jpg"; // 吴彦祖头像

// 添加更多真实头像URL供AI玩家使用
const REAL_AVATARS = {
    male: [
        "https://randomuser.me/api/portraits/men/32.jpg",
        "https://randomuser.me/api/portraits/men/51.jpg",
        "https://randomuser.me/api/portraits/men/61.jpg",
        "https://randomuser.me/api/portraits/men/72.jpg",
        "https://randomuser.me/api/portraits/men/83.jpg",
        "https://randomuser.me/api/portraits/men/4.jpg",
        "https://randomuser.me/api/portraits/men/15.jpg",
        "https://randomuser.me/api/portraits/men/26.jpg",
        "https://randomuser.me/api/portraits/men/37.jpg",
        "https://randomuser.me/api/portraits/men/48.jpg"
    ],
    female: [
        "https://randomuser.me/api/portraits/women/32.jpg",
        "https://randomuser.me/api/portraits/women/51.jpg",
        "https://randomuser.me/api/portraits/women/61.jpg",
        "https://randomuser.me/api/portraits/women/72.jpg",
        "https://randomuser.me/api/portraits/women/83.jpg",
        "https://randomuser.me/api/portraits/women/4.jpg",
        "https://randomuser.me/api/portraits/women/15.jpg",
        "https://randomuser.me/api/portraits/women/26.jpg",
        "https://randomuser.me/api/portraits/women/37.jpg",
        "https://randomuser.me/api/portraits/women/48.jpg"
    ]
};

// 修改getPlayerAvatar函数，为主玩家使用吴彦祖头像
function getPlayerAvatar(index, gender, style = 'realistic') {
    // 如果是主玩家 (index=0)，返回吴彦祖头像
    if (index === 0) {
        return MAIN_PLAYER_AVATAR;
    }
    
    // 使用预设的玩家数据库
    if (index < POKER_PLAYERS_DATABASE.length) {
        const player = POKER_PLAYERS_DATABASE[index];
        // 使用数据库中的头像，如果是有效的URL
        if (player.avatarUrl && player.avatarUrl.startsWith('http')) {
            return player.avatarUrl;
        }
    }
    
    // 如果要求使用真实风格头像或有90%的概率使用真实头像
    if (style === 'realistic' || Math.random() < 0.9) {
        // 根据性别选择真实头像
        const genderAvatars = REAL_AVATARS[gender || (Math.random() > 0.5 ? 'male' : 'female')];
        // 添加一些随机性，避免每次游戏相同位置的AI使用相同头像
        const avatarIndex = (index + Math.floor(Math.random() * 100)) % genderAvatars.length;
        return genderAvatars[avatarIndex];
    }
    
    // 卡通/抽象风格使用现有逻辑
    if (style === 'cartoon' && FALLBACK_AVATARS.cartoon.length > 0) {
        const cartoonIndex = (index + Math.floor(Math.random() * 100)) % FALLBACK_AVATARS.cartoon.length;
        return FALLBACK_AVATARS.cartoon[cartoonIndex];
    }
    
    if (style === 'abstract' && FALLBACK_AVATARS.abstract.length > 0) {
        const abstractIndex = (index + Math.floor(Math.random() * 100)) % FALLBACK_AVATARS.abstract.length;
        return FALLBACK_AVATARS.abstract[abstractIndex];
    }
    
    // 默认随机选择真实头像
    const randomGender = Math.random() > 0.5 ? 'male' : 'female';
    const genderAvatars = REAL_AVATARS[randomGender];
    const avatarIndex = (index + Math.floor(Math.random() * 100)) % genderAvatars.length;
    return genderAvatars[avatarIndex];
}

// 开始游戏
function startGame(playerName) {
    console.log("开始游戏，玩家名称:", playerName);
    
    // 隐藏欢迎界面，显示游戏界面
    welcomeScreen.style.display = 'none';
    gameScreen.style.display = 'block';
    
    // 更新声音设置
    gameState.soundEnabled = enableSoundCheckbox.checked;
    
    // 根据游戏模式设置
    if (gameState.tournament.isEnabled) {
        // 显示锦标赛信息区域
        document.getElementById('tournamentInfo').style.display = 'block';
        document.getElementById('cashGameInfo').style.display = 'none';
    } else {
        // 显示现金游戏信息
        document.getElementById('cashGameInfo').style.display = 'block';
        document.getElementById('tournamentInfo').style.display = 'none';
        
        // 设置现金游戏盲注
        gameState.smallBlind = 5;
        gameState.bigBlind = 10;
        
        // 更新盲注显示
        document.getElementById('cashGameBlinds').textContent = `${gameState.smallBlind}/${gameState.bigBlind}`;
    }
    
    // 获取玩家初始筹码数量
    let startingChips = 1000; // 默认现金游戏筹码
    if (gameState.tournament.isEnabled) {
        startingChips = gameState.tournament.startingChips;
    }
    
    // 为主玩家获取真实头像
    const playerAvatar = getPlayerAvatar(0);
    
    // 创建玩家
    const mainPlayer = {
        id: 0,
        name: playerName,
        chips: startingChips,
        hand: [],
        bet: 0,
        folded: false,
        isAllIn: false,
        avatarColor: generateAvatarColor(0),
        avatar: playerAvatar,
        eliminated: false
    };
    
    // 重置游戏状态
    gameState.players = [mainPlayer];
    gameState.gamePhase = 'waiting';
    gameState.pot = 0;
    gameState.currentBet = 0;
    gameState.dealerPosition = 0;
    gameState.roundNumber = 1;
    
    // 更新显示
    playerNameDisplay.textContent = playerName;
    playerChipsDisplay.textContent = mainPlayer.chips;
    gamePhaseDisplay.textContent = PHASE_DESCRIPTIONS.waiting;
    
    // 添加AI对手
    addAIPlayers(startingChips);
    
    // 渲染玩家位置
    renderPlayers();
    
    // 显示荷官按钮
    dealerButton.style.display = 'flex';
    
    // 清空历史记录
    clearHistory();
    
    // 更新回合数显示
    roundNumberDisplay.textContent = gameState.roundNumber;
    
    // 添加游戏开始记录
    if (gameState.tournament.isEnabled) {
        addHistoryItem('系统', '锦标赛开始', null, true);
        initializeTournament();
    } else {
        addHistoryItem('系统', '游戏开始', null, true);
    }
    
    // 显示并启用"开始牌局"按钮
    const startRoundBtn = document.getElementById('startRoundBtn');
    startRoundBtn.classList.remove('hidden');
    startRoundBtn.disabled = false;
    
    // 禁用游戏按钮
    foldBtn.disabled = true;
    checkBtn.disabled = true;
    callBtn.disabled = true;
    raiseBtn.disabled = true;
    allInBtn.disabled = true;
    
    // 添加提示文本
    addHistoryItem('系统', '请点击"开始牌局"按钮开始游戏', null, true);
    
    console.log("游戏初始化完成，等待开始牌局");
}

// 添加AI玩家
function addAIPlayers(startingChips = 1000) {
    // 确保总玩家数为7人（1个主玩家+6个AI玩家）
    const numAIPlayers = 6;
    
    // 从数据库随机选择AI玩家
    const shuffledPlayers = [...POKER_PLAYERS_DATABASE].sort(() => Math.random() - 0.5);
    
    for (let i = 0; i < numAIPlayers; i++) {
        // 如果有足够的玩家数据，使用数据库中的玩家
        if (i < shuffledPlayers.length) {
            const playerData = shuffledPlayers[i];
            
            // 确定AI的难度级别（根据玩家风格决定）
            let aiLevel = 'normal';
            let avatarStyle = 'realistic'; // 使用真实头像风格
            
            if (playerData.style.includes('激进') || playerData.style.includes('松凶')) {
                aiLevel = 'aggressive';
            } else if (playerData.style.includes('稳健') || playerData.style.includes('紧')) {
                aiLevel = 'conservative';
            } else if (playerData.style.includes('创新') || playerData.style.includes('不按常理')) {
                // 创新型玩家有10%概率使用非真实头像
                if (Math.random() < 0.1) {
                    avatarStyle = Math.random() < 0.5 ? 'cartoon' : 'abstract';
                }
            }
            
            // 根据玩家类型和风格获取头像
            const aiAvatar = getPlayerAvatar(i + 1, playerData.gender, avatarStyle);
            
            gameState.players.push({
                id: i + 1,
                name: playerData.name,
                chips: startingChips,
                hand: [],
                bet: 0,
                folded: false,
                isAllIn: false,
                avatarColor: generateAvatarColor(i + 1),
                avatar: aiAvatar,
                eliminated: false,
                aiLevel: aiLevel,
                nickname: playerData.nickname || '',
                country: playerData.country || '',
                characteristic: playerData.characteristic || '',
                avatarStyle: avatarStyle // 保存头像风格信息
            });
        } else {
            // 如果数据库玩家不足，使用旧的方法创建AI玩家
            const aiNames = ['AI玩家' + (i + 1)];
            const aiName = aiNames[0];
            const gender = Math.random() > 0.5 ? 'male' : 'female';
            
            // 使用真实风格头像
            const aiAvatar = getPlayerAvatar(i + 1, gender, 'realistic');
        
        gameState.players.push({
            id: i + 1,
            name: aiName,
            chips: startingChips,
            hand: [],
            bet: 0,
            folded: false,
            isAllIn: false,
            avatarColor: generateAvatarColor(i + 1),
                avatar: aiAvatar,
                eliminated: false,
                aiLevel: 'normal',
                avatarStyle: 'realistic' // 保存头像风格信息
        });
        }
    }
}

// 渲染玩家位置
function renderPlayers() {
    playersContainer.innerHTML = '';
    
    gameState.players.forEach((player, index) => {
        if (index > 0) { // 不渲染主玩家，因为主玩家在底部有特殊区域
            const playerElement = document.createElement('div');
            playerElement.className = 'player-position';
            playerElement.id = `player${player.id}`;
            playerElement.style.top = PLAYER_POSITIONS[index].top;
            playerElement.style.left = PLAYER_POSITIONS[index].left;
            
            // 添加弃牌标记
            if (player.folded) {
                playerElement.classList.add('folded');
            }
            
            // 根据ALL IN状态添加类名
            if (player.isAllIn) {
                playerElement.classList.add('all-in');
            }
            
            // 根据淘汰状态添加类名
            if (player.eliminated) {
                playerElement.classList.add('eliminated');
            }
            
            // 显示玩家昵称和国籍（如果有）
            let playerInfo = player.name;
            if (player.nickname && player.country) {
                playerInfo = `${player.name} <span class="player-nickname">"${player.nickname}"</span> <span class="player-country">${player.country}</span>`;
            } else if (player.nickname) {
                playerInfo = `${player.name} <span class="player-nickname">"${player.nickname}"</span>`;
            } else if (player.country) {
                playerInfo = `${player.name} <span class="player-country">${player.country}</span>`;
            }
            
            // 根据AI级别添加风格类
            let aiLevelClass = '';
            if (player.aiLevel === 'aggressive') {
                aiLevelClass = 'aggressive-player';
            } else if (player.aiLevel === 'conservative') {
                aiLevelClass = 'conservative-player';
            }
            
            // 添加头像风格类
            let avatarStyleClass = '';
            if (player.avatarStyle === 'cartoon') {
                avatarStyleClass = 'cartoon-avatar';
            } else if (player.avatarStyle === 'abstract') {
                avatarStyleClass = 'abstract-avatar';
            }
            
            // 创建头像HTML
            const avatarHtml = `
                <div class="player-avatar ${aiLevelClass} ${avatarStyleClass}" style="background-color: ${player.avatarColor}">
                    <img src="${player.avatar}" alt="${player.name}" class="avatar-image" onerror="this.classList.add('error'); this.parentElement.classList.add('fallback-active');">
                    <div class="avatar-image-fallback">${player.name.charAt(0).toUpperCase()}</div>
                    ${player.aiLevel === 'aggressive' ? '<div class="ai-badge aggressive-badge">激进型</div>' : ''}
                    ${player.aiLevel === 'conservative' ? '<div class="ai-badge conservative-badge">保守型</div>' : ''}
                </div>
                <div class="player-name ${aiLevelClass}">${playerInfo}</div>
                <div class="player-chips">
                    <img src="assets/images/chip.svg" alt="筹码" class="chip-icon">
                    <span>${player.chips}</span>
                </div>
                ${player.isAllIn ? '<div class="player-status all-in-badge">ALL IN</div>' : ''}
                ${player.eliminated ? '<div class="player-status eliminated-badge">已淘汰</div>' : ''}
            `;
            
            playerElement.innerHTML = avatarHtml;
            playersContainer.appendChild(playerElement);
        }
    });
    
    // 更新主玩家头像
    updateMainPlayerAvatar();
}

// 更新主玩家头像
function updateMainPlayerAvatar() {
    if (gameState.players.length === 0) return;
    
    const mainPlayer = gameState.players[0];
    const mainPlayerAvatar = document.createElement('div');
    mainPlayerAvatar.className = 'main-player-avatar';
    
    // 添加头像风格类
    if (mainPlayer.avatarStyle === 'cartoon') {
        mainPlayerAvatar.classList.add('cartoon-avatar');
    } else if (mainPlayer.avatarStyle === 'abstract') {
        mainPlayerAvatar.classList.add('abstract-avatar');
    }
    
    // 添加主玩家状态
    let statusHTML = '';
    if (mainPlayer.folded) {
        statusHTML = '<div class="player-status folded-badge">已弃牌</div>';
    } else if (mainPlayer.isAllIn) {
        statusHTML = '<div class="player-status all-in-badge">ALL IN</div>';
    } else if (mainPlayer.eliminated) {
        statusHTML = '<div class="player-status eliminated-badge">已淘汰</div>';
    }
    
    mainPlayerAvatar.innerHTML = `
        <div class="avatar-container">
            <img src="${mainPlayer.avatar}" alt="${mainPlayer.name}" class="avatar-image" onerror="this.classList.add('error'); this.parentElement.classList.add('fallback-active');">
            <div class="avatar-image-fallback">${mainPlayer.name.charAt(0).toUpperCase()}</div>
        </div>
        ${statusHTML}
    `;
    
    // 查找并更新主玩家头像容器
    const existingAvatar = document.querySelector('.main-player-avatar');
    if (existingAvatar) {
        existingAvatar.remove();
    }
    
    // 添加头像到主玩家区域
    const playerHand = document.querySelector('.player-hand');
    if (playerHand) {
        playerHand.insertAdjacentElement('afterend', mainPlayerAvatar);
    }
}

// 创建新牌组
function createDeck() {
    const deck = [];
    SUITS.forEach(suit => {
        CARD_VALUES.forEach(value => {
            deck.push({ suit, value });
        });
    });
    return deck;
}

// 洗牌
function shuffle(deck) {
    for (let i = deck.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [deck[i], deck[j]] = [deck[j], deck[i]];
    }
    return deck;
}

// 开始新一轮
function startNewRound() {
    console.log("开始新一轮");
    
    // 清除所有玩家的手牌显示
    document.querySelectorAll('.player-cards-display').forEach(el => {
        el.remove();
    });
    
    // 锦标赛模式 - 检查玩家淘汰
    if (gameState.tournament.isEnabled) {
        // 检查每个玩家是否被淘汰
        let eliminatedThisRound = false;
        
        for (let i = 0; i < gameState.players.length; i++) {
            const player = gameState.players[i];
            if (player.chips <= 0 && !player.eliminated) {
                player.eliminated = true;
                handlePlayerElimination(i);
                eliminatedThisRound = true;
            }
        }
        
        // 如果有玩家被淘汰，更新界面
        if (eliminatedThisRound) {
            renderPlayers();
            updateTournamentDisplay();
        }
        
        // 如果锦标赛已结束，不继续游戏
        if (gameState.tournament.isEnabled && gameState.tournament.isFinished) {
            console.log("锦标赛已结束，不再开始新回合");
            return;
        }
    }
    
    // 重置游戏状态
    console.log("重置游戏状态");
    gameState.deck = shuffle(createDeck());
    gameState.communityCards = [];
    gameState.pot = 0;
    gameState.currentBet = 0;
    gameState.gamePhase = 'preflop';
    
    // 更新游戏阶段显示
    updateGamePhaseDisplay();
    console.log("游戏阶段更新为:", gameState.gamePhase);
    
    // 移动荷官按钮（轮换位置）
    gameState.dealerPosition = (gameState.dealerPosition + 1) % gameState.players.length;
    updateDealerButton(gameState.dealerPosition);
    console.log("荷官位置更新为:", gameState.dealerPosition);
    
    // 寻找下一个未淘汰的玩家作为庄家
    if (gameState.tournament.isEnabled) {
        let attempts = 0;
        while (attempts < gameState.players.length) {
            if (gameState.players[gameState.dealerPosition].eliminated) {
                gameState.dealerPosition = (gameState.dealerPosition + 1) % gameState.players.length;
                attempts++;
            } else {
                break;
            }
        }
        updateDealerButton(gameState.dealerPosition);
        console.log("锦标赛模式 - 更新后的荷官位置:", gameState.dealerPosition);
    }
    
    // 计算小盲和大盲的位置
    let smallBlindPos = (gameState.dealerPosition + 1) % gameState.players.length;
    let bigBlindPos = (gameState.dealerPosition + 2) % gameState.players.length;
    console.log("小盲位置:", smallBlindPos, "大盲位置:", bigBlindPos);
    
    // 在锦标赛模式下检查小盲和大盲是否已淘汰
    if (gameState.tournament.isEnabled) {
        // 寻找有效的小盲位置
        let attempts = 0;
        while (attempts < gameState.players.length) {
            if (gameState.players[smallBlindPos].eliminated) {
                smallBlindPos = (smallBlindPos + 1) % gameState.players.length;
                attempts++;
            } else {
                break;
            }
        }
        
        // 寻找有效的大盲位置
        attempts = 0;
        bigBlindPos = (smallBlindPos + 1) % gameState.players.length;
        while (attempts < gameState.players.length) {
            if (gameState.players[bigBlindPos].eliminated) {
                bigBlindPos = (bigBlindPos + 1) % gameState.players.length;
                attempts++;
            } else {
                break;
            }
        }
        console.log("锦标赛模式 - 更新后的盲注位置 - 小盲:", smallBlindPos, "大盲:", bigBlindPos);
    }
    
    // 重置玩家状态
    console.log("重置玩家状态");
    gameState.players.forEach(player => {
        player.hand = [];
        player.bet = 0;
        player.folded = false;
        player.isAllIn = false;
    });
    
    // 清空卡牌显示
    console.log("清空卡牌显示");
    playerCardElements.forEach(element => {
        element.textContent = '';
        element.className = 'card-placeholder';
        element.style.backgroundColor = 'rgba(255, 255, 255, 0.1)';
    });
    
    communityCardElements.forEach(element => {
        element.textContent = '';
        element.className = 'card-placeholder';
        element.style.backgroundColor = 'rgba(255, 255, 255, 0.1)';
    });
    
    // 发两张手牌给每个玩家
    console.log("开始发牌");
    gameState.players.forEach(player => {
        // 跳过已淘汰的玩家
        if (gameState.tournament.isEnabled && player.eliminated) {
            return;
        }
        
        player.hand = [
            gameState.deck.pop(),
            gameState.deck.pop()
        ];
    });
    
    // 显示主玩家的手牌
    renderPlayerHand();
    console.log("主玩家手牌已显示");
    
    // 小盲下注
    const smallBlindAmount = gameState.smallBlind;
    gameState.players[smallBlindPos].chips -= smallBlindAmount;
    gameState.players[smallBlindPos].bet = smallBlindAmount;
    gameState.pot += smallBlindAmount;
    console.log("小盲下注完成:", smallBlindAmount);
    
    // 创建小盲通知
    createActionNotification(
        `小盲 $${smallBlindAmount}`,
        {
            left: PLAYER_POSITIONS[smallBlindPos].left,
            top: PLAYER_POSITIONS[smallBlindPos].top
        }
    );
    
    // 小盲筹码动画
    createChipAnimation(
        {
            left: PLAYER_POSITIONS[smallBlindPos].left,
            top: PLAYER_POSITIONS[smallBlindPos].top
        },
        { left: '50%', top: '50%' },
        smallBlindAmount
    );
    
    // 添加小盲记录
    addHistoryItem(gameState.players[smallBlindPos].name, 'smallBlind', smallBlindAmount);
    
    // 大盲下注
    const bigBlindAmount = gameState.bigBlind;
    gameState.players[bigBlindPos].chips -= bigBlindAmount;
    gameState.players[bigBlindPos].bet = bigBlindAmount;
    gameState.pot += bigBlindAmount;
    gameState.currentBet = bigBlindAmount;
    console.log("大盲下注完成:", bigBlindAmount);
    
    // 创建大盲通知
    createActionNotification(
        `大盲 $${bigBlindAmount}`,
        {
            left: PLAYER_POSITIONS[bigBlindPos].left,
            top: PLAYER_POSITIONS[bigBlindPos].top
        }
    );
    
    // 大盲筹码动画
    createChipAnimation(
        {
            left: PLAYER_POSITIONS[bigBlindPos].left,
            top: PLAYER_POSITIONS[bigBlindPos].top
        },
        { left: '50%', top: '50%' },
        bigBlindAmount
    );
    
    // 添加大盲记录
    addHistoryItem(gameState.players[bigBlindPos].name, 'bigBlind', bigBlindAmount);
    
    // 更新显示
    updateGameDisplay();
    console.log("游戏显示已更新");
    
    // 更新回合数
    gameState.roundNumber++;
    roundNumberDisplay.textContent = gameState.roundNumber;
    console.log("回合数更新为:", gameState.roundNumber);
    
    // 添加新回合记录
    addHistoryItem('系统', `第 ${gameState.roundNumber} 回合开始`, null, true);
    
    // 延迟一下，让玩家看到小盲大盲的下注
    console.log("等待1.5秒后发牌");
    setTimeout(() => {
        // 发牌给所有玩家
        dealCards();
        console.log("发牌完成");
        
        // 再延迟一下，让玩家看到发牌动画
    setTimeout(() => {
        // 开始回合 - 从大盲后面的玩家开始行动
            console.log("开始下注回合");
        startBettingRound();
        }, 1500);
    }, 1500);
}

// 发牌
function dealCards() {
    // 每个玩家发2张牌
    
    // 特殊处理：如果主玩家名字是"王总"，给予好牌
    const mainPlayer = gameState.players[0];
    if (mainPlayer.name === "王总") {
        // 为王总准备好牌
        const goodStartingHands = [
            // 高对子
            [{ suit: '♠', value: 'A' }, { suit: '♥', value: 'A' }],
            [{ suit: '♠', value: 'K' }, { suit: '♥', value: 'K' }],
            [{ suit: '♠', value: 'Q' }, { suit: '♥', value: 'Q' }],
            // AK, AQ
            [{ suit: '♠', value: 'A' }, { suit: '♠', value: 'K' }],
            [{ suit: '♦', value: 'A' }, { suit: '♦', value: 'Q' }],
            // 同花
            [{ suit: '♥', value: 'J' }, { suit: '♥', value: 'Q' }],
            [{ suit: '♣', value: '10' }, { suit: '♣', value: 'J' }]
        ];
        
        // 随机选择一套好牌
        const selectedHand = goodStartingHands[Math.floor(Math.random() * goodStartingHands.length)];
        
        // 从牌堆中移除这些牌
        selectedHand.forEach(card => {
            const cardIndex = gameState.deck.findIndex(c => c.suit === card.suit && c.value === card.value);
            if (cardIndex !== -1) {
                gameState.deck.splice(cardIndex, 1);
            }
        });
        
        // 为王总发牌
        mainPlayer.hand = [...selectedHand];
        
        // 为其他玩家发牌
        for (let i = 0; i < 2; i++) {
            gameState.players.forEach((player, playerIndex) => {
                if (playerIndex > 0) { // 跳过主玩家
                    setTimeout(() => {
                        player.hand.push(gameState.deck.pop());
                    }, 300 * (i * (gameState.players.length - 1) + playerIndex));
                }
            });
        }
        
        // 为主玩家显示牌
        setTimeout(() => {
            renderPlayerHand(0);
            playSound('cardDeal');
            
            setTimeout(() => {
                renderPlayerHand(1);
                playSound('cardDeal');
            }, 300);
        }, 300);
    } else {
        // 普通发牌流程
        for (let i = 0; i < 2; i++) {
            gameState.players.forEach((player, playerIndex) => {
                setTimeout(() => {
                    player.hand.push(gameState.deck.pop());
                    
                    // 为主玩家的牌添加发牌动画
                    if (playerIndex === 0) {
                        renderPlayerHand(i);
                        playSound('cardDeal');
                    }
                }, 300 * (i * gameState.players.length + playerIndex));
            });
        }
    }
}

// 渲染玩家手牌
function renderPlayerHand(cardIndex = null) {
    const mainPlayer = gameState.players[0];
    
    if (cardIndex !== null) {
        // 单张牌动画
        const card = mainPlayer.hand[cardIndex];
        const element = playerCardElements[cardIndex];
        element.textContent = `${card.value}${card.suit}`;
        element.style.backgroundColor = '#fff';
        element.style.color = ['♥', '♦'].includes(card.suit) ? 'red' : 'black';
        element.classList.add('dealt');
    } else {
        // 显示所有牌
        mainPlayer.hand.forEach((card, index) => {
            const element = playerCardElements[index];
            element.textContent = `${card.value}${card.suit}`;
            element.style.backgroundColor = '#fff';
            element.style.color = ['♥', '♦'].includes(card.suit) ? 'red' : 'black';
        });
    }
}

// 渲染公共牌
function renderCommunityCards() {
    communityCardElements.forEach((element, index) => {
        if (index < gameState.communityCards.length) {
            const card = gameState.communityCards[index];
            element.textContent = `${card.value}${card.suit}`;
            element.style.backgroundColor = '#fff';
            element.style.color = ['♥', '♦'].includes(card.suit) ? 'red' : 'black';
            element.classList.add('dealt');
            playSound('cardDeal');
        } else {
            element.textContent = '';
            element.style.backgroundColor = 'rgba(255, 255, 255, 0.1)';
            element.classList.remove('dealt');
        }
    });
}

// 更新游戏显示
function updateGameDisplay() {
    const mainPlayer = gameState.players[0];
    playerChipsDisplay.textContent = mainPlayer.chips;
    potDisplay.textContent = `底池: $${gameState.pot}`;
    
    // 更新bet slider最大值
    betSlider.max = mainPlayer.chips;
    betSlider.value = gameState.currentBet > 0 ? gameState.currentBet : 1;
    updateBetAmount();
    
    // 更新按钮状态
    updateButtonStates();
    
    // 更新AI玩家状态
    renderPlayers();
    
    // 更新筹码显示
    updateChipsDisplay();
    
    // 更新游戏阶段
    updateGamePhaseDisplay();
    
    // 更新当前玩家
    updateCurrentPlayerDisplay();
}

// 更新下注金额显示
function updateBetAmount() {
    betAmount.textContent = `$${betSlider.value}`;
}

// 更新按钮状态
function updateButtonStates() {
    const mainPlayer = gameState.players[0];
    const canCheck = gameState.currentBet === mainPlayer.bet;
    
    // 如果玩家已弃牌或全押，禁用所有按钮
    if (mainPlayer.folded || mainPlayer.isAllIn || mainPlayer.chips === 0) {
        foldBtn.disabled = true;
        checkBtn.disabled = true;
        callBtn.disabled = true;
        raiseBtn.disabled = true;
        allInBtn.disabled = true;
        return;
    }
    
    checkBtn.disabled = !canCheck;
    callBtn.disabled = canCheck;
    raiseBtn.disabled = mainPlayer.chips + mainPlayer.bet <= gameState.currentBet;
    allInBtn.disabled = mainPlayer.chips === 0;
    
    // 显示跟注金额
    const callAmount = gameState.currentBet - mainPlayer.bet;
    callBtn.textContent = `跟注 $${callAmount}`;
    
    // 设置最小加注额
    const minRaise = Math.max(gameState.currentBet * 2, mainPlayer.bet + 1);
    betSlider.min = Math.min(minRaise, mainPlayer.chips + mainPlayer.bet);
    betSlider.max = mainPlayer.chips + mainPlayer.bet;
    betSlider.value = Math.min(minRaise, mainPlayer.chips + mainPlayer.bet);
    updateBetAmount();
}

// 开始下注回合
function startBettingRound() {
    console.log("开始下注回合, 当前游戏阶段:", gameState.gamePhase);
    let startPosition;
    
    if (gameState.gamePhase === 'preflop') {
        // 前翻牌圈从大盲注的下一个位置开始
        startPosition = (gameState.dealerPosition + 3) % gameState.players.length;
        console.log("前翻牌圈，起始位置:", startPosition);
    } else {
        // 其他轮次从小盲开始，即庄家的下一个位置
        startPosition = (gameState.dealerPosition + 1) % gameState.players.length;
        console.log("其他轮次，起始位置:", startPosition);
    }
    
    // 计算有多少玩家仍然活跃
    const activePlayers = gameState.players.filter(p => !p.folded && !p.isAllIn && p.chips > 0);
    console.log("活跃玩家数量:", activePlayers.length);
    
    // 如果只有一个活跃玩家或没有活跃玩家，直接进入下一阶段
    if (activePlayers.length <= 1) {
        console.log("活跃玩家数量不足，直接进入下一阶段");
        advanceGamePhase();
        return;
    }
    
    // 重置所有玩家的当前下注金额
    if (gameState.gamePhase !== 'preflop') {
        gameState.currentBet = 0;
        gameState.players.forEach(player => {
            player.bet = 0;
        });
        console.log("当前阶段非preflop，重置所有玩家下注");
    }
    
    // 寻找起始位置的下一个有效玩家
    let currentPos = startPosition;
    let found = false;
    
    // 遍历一圈寻找有效的起始玩家
    for (let i = 0; i < gameState.players.length; i++) {
        const playerIndex = (currentPos + i) % gameState.players.length;
        const player = gameState.players[playerIndex];
        
        if (!player.folded && !player.isAllIn && player.chips > 0) {
            gameState.currentPlayer = playerIndex;
            console.log(`找到有效起始玩家: ${player.name}，索引: ${playerIndex}`);
            found = true;
            break;
        }
    }
    
    // 如果没找到有效玩家，直接进入下一阶段
    if (!found) {
        console.log("没有找到有效的起始玩家，直接进入下一阶段");
        advanceGamePhase();
        return;
    }
    
    // 更新当前玩家显示
    updateCurrentPlayerDisplay();
    console.log("当前玩家显示已更新");
    
    // 更新历史记录中的当前行动玩家
    updateCurrentActionPlayer(gameState.currentPlayer);
    console.log("历史记录中的当前行动玩家已更新");
    
    // 如果当前玩家是主玩家
    if (gameState.currentPlayer === 0) {
        console.log("当前轮到主玩家行动");
        // 思考气泡
        const mainPlayer = gameState.players[0];
        
        // 正确传递玩家名称
        const thoughtText = getRandomDialog('thinking', mainPlayer.name);
        createDialogBubble(thoughtText, PLAYER_POSITIONS[0], 0);
        
        // 启用控制按钮
        foldBtn.disabled = false;
        checkBtn.disabled = false;
        callBtn.disabled = false;
        raiseBtn.disabled = false;
        allInBtn.disabled = false;
        console.log("主玩家控制按钮已启用");
        
        // 更新按钮状态
        updateButtonStates();
        console.log("按钮状态已更新");
        
        // 设置行动计时器
        setTimeout(() => {
            console.log("为主玩家启动行动计时器");
        startActionTimer();
        }, 300);
    } else {
        console.log(`当前轮到AI玩家(${gameState.players[gameState.currentPlayer].name})行动`);
        // 先设置计时器，然后AI玩家行动
        setTimeout(() => {
            console.log("为AI玩家启动行动计时器");
            startActionTimer();
        simulateAIActions();
        }, 500);
    }
}

// 设置行动计时器
function startActionTimer() {
    console.log("启动计时器，当前玩家:", gameState.players[gameState.currentPlayer].name);
    
    // 获取当前玩家
    const currentPlayer = gameState.players[gameState.currentPlayer];
    
    // 清除已有计时器
    clearActionTimer();
    
    // 设置时间
    let timeLeft = gameState.settings.actionTimeLimit || DEFAULT_ACTION_TIME;
    console.log(`设置计时器时间:${timeLeft}秒`);
    
    // 创建计时器显示
    const timerElement = document.createElement('div');
    timerElement.id = 'actionTimer';
    timerElement.className = 'action-timer';
    
    // 设置计时器位置
    const timerPos = PLAYER_POSITIONS[gameState.currentPlayer];
    timerElement.style.top = timerPos.top;
    timerElement.style.left = timerPos.left;
    
    // 添加到界面
    document.body.appendChild(timerElement);
        
        // 更新计时器显示
            timerElement.textContent = timeLeft;
            
    // 添加警告音效
    const warningTimes = [10, 5, 3]; // 倒计时10秒、5秒和3秒时播放警告音
    gameState.timeWarningTimers = [];
    
    // 为每个警告时间设置计时器
    warningTimes.forEach(warningTime => {
        if (timeLeft > warningTime) {
            const warningDelay = (timeLeft - warningTime) * 1000;
            const warningTimer = setTimeout(() => {
                // 播放警告音效
                playSound('timeWarning');
                
                // 添加视觉提示 - 计时器闪烁
                timerElement.classList.add('timer-warning');
                
                // 如果是3秒警告，添加计时器脉动动画
                if (warningTime <= 3) {
                    timerElement.classList.add('timer-pulse');
                }
                
                // 显示警告消息
                createActionNotification(`剩余${warningTime}秒!`, { 
                    left: timerPos.left, 
                    top: timerPos.top
                });
            }, warningDelay);
            
            // 保存计时器引用以便后续清除
            gameState.timeWarningTimers.push(warningTimer);
        }
    });
    
    // 创建主计时器
    gameState.actionTimer = setInterval(() => {
        // 减少剩余时间
        timeLeft--;
        
        // 更新显示
        timerElement.textContent = timeLeft;
        
        // 时间到了
        if (timeLeft <= 0) {
            // 清除计时器
            clearInterval(gameState.actionTimer);
            gameState.actionTimer = null;
            
            // 移除计时器显示
            timerElement.remove();
            
            console.log("时间到，为玩家自动执行操作");
            
            // 根据玩家类型执行默认操作
            if (gameState.currentPlayer === 0) {
                // 主玩家超时处理
                // 禁用按钮
                foldBtn.disabled = true;
                checkBtn.disabled = true;
                callBtn.disabled = true;
                raiseBtn.disabled = true;
                allInBtn.disabled = true;
                
                // 判断是否可以让牌
                const canCheck = gameState.currentBet === gameState.players[0].bet;
                
                // 创建超时通知
                createActionNotification('操作超时!', { left: '50%', top: '70%' });
                
                if (canCheck) {
                    // 如果可以让牌，则让牌
                    console.log('操作超时: 自动让牌');
                    setTimeout(() => handleCheck(), 500);
            } else {
                    // 否则弃牌
                    console.log('操作超时: 自动弃牌');
                    setTimeout(() => handleFold(), 500);
                }
            } else {
                // AI玩家超时处理
                console.log('AI玩家超时，模拟随机操作');
                const aiPlayer = gameState.players[gameState.currentPlayer];
                const canCheck = gameState.currentBet === aiPlayer.bet;
                
                if (canCheck) {
                    aiPlayer.folded = false;
                    addHistoryItem(aiPlayer.name, 'check');
                    showPlayerDialog(aiPlayer.id, 'check');
                    playSound('check');
                } else {
                    aiPlayer.folded = true;
                    addHistoryItem(aiPlayer.name, 'fold');
                    showPlayerDialog(aiPlayer.id, 'fold');
                    playSound('fold');
                }
                
                // 添加超时通知
                createActionNotification('AI操作超时', { 
                    left: PLAYER_POSITIONS[gameState.currentPlayer].left, 
                    top: PLAYER_POSITIONS[gameState.currentPlayer].top 
                });
                
                // 进入下一个阶段
                setTimeout(() => {
                    afterPlayerAction();
                }, 1000);
            }
        }
    }, 1000);
}

// 清除行动计时器
function clearActionTimer() {
    console.log("清除行动计时器");
    
    // 清除主计时器
    if (gameState.actionTimer) {
        clearInterval(gameState.actionTimer);
        gameState.actionTimer = null;
        console.log("主计时器已清除");
    }
    
    // 清除警告音计时器
    if (gameState.timeWarningTimers && gameState.timeWarningTimers.length > 0) {
        gameState.timeWarningTimers.forEach(timer => {
            clearTimeout(timer);
            console.log("警告音计时器已清除");
        });
        gameState.timeWarningTimers = [];
    }
    
    // 移除计时器显示
    const timerElement = document.getElementById('actionTimer');
    if (timerElement) {
        timerElement.remove();
        console.log("计时器显示已移除");
    } else {
        console.log("未找到计时器显示元素");
    }
}

// 模拟AI行动
function simulateAIActions() {
    console.log("AI准备行动，玩家:", gameState.players[gameState.currentPlayer].name);
    
    // 获取当前AI玩家
    const aiPlayer = gameState.players[gameState.currentPlayer];
    
    // 如果已经弃牌或ALL IN，跳过行动
    if (aiPlayer.folded || aiPlayer.isAllIn) {
        console.log("AI玩家已弃牌或ALL IN，跳过行动");
        moveToNextPlayer();
        return;
    }
    
    // 显示思考动画
    displayThinkingAnimation(aiPlayer.id);
    
    // 延迟一些时间再做出决策 (让游戏节奏更自然)
    setTimeout(() => {
        // 使用新的AI决策逻辑
        const decision = makeAIDecision(gameState.currentPlayer);
        console.log(`AI决策结果: ${decision}`);
        
        // 根据决策执行相应操作
        switch (decision) {
            case 'fold':
                // AI弃牌
                aiPlayer.folded = true;
                addHistoryItem(aiPlayer.name, 'fold');
                
                // 显示弃牌对话
                showPlayerDialog(aiPlayer.id, 'fold');
                
                // 播放弃牌音效
                playSound('fold');
                
                break;
                
            case 'check':
                // AI让牌
                addHistoryItem(aiPlayer.name, 'check');
                
                // 显示让牌对话
                showPlayerDialog(aiPlayer.id, 'check');
                
                // 播放让牌音效
                        playSound('check');
                        
                    break;
                
            case 'call':
                // 计算跟注金额
                const callAmount = gameState.currentBet - aiPlayer.bet;
                
                // 确保跟注不超过剩余筹码
                const actualCallAmount = Math.min(callAmount, aiPlayer.chips);
                
                // 更新玩家筹码和下注
                aiPlayer.chips -= actualCallAmount;
                aiPlayer.bet += actualCallAmount;
                
                // 更新底池
                gameState.pot += actualCallAmount;
                        
                        // 添加历史记录
                addHistoryItem(aiPlayer.name, 'call', actualCallAmount);
                        
                // 显示跟注对话
                showPlayerDialog(aiPlayer.id, 'call');
                
                // 创建筹码动画
                        createChipAnimation(
                    PLAYER_POSITIONS[aiPlayer.id],
                    { top: '50%', left: '50%' },
                    actualCallAmount
                );
                
                // 播放跟注音效
                playSound('call');
                
                        break;
                
            case 'raise':
                // 设置一个合理的加注额 (当前下注的1.5-2.5倍或剩余筹码的1/3，取较小值)
                const minRaise = gameState.currentBet * 2;
                const maxRaise = Math.min(gameState.currentBet * 3, aiPlayer.chips + aiPlayer.bet);
                const aiRaiseTarget = Math.floor(minRaise + Math.random() * (maxRaise - minRaise));
                
                // 需要先支付当前下注与玩家已下注的差额
                const additionalBet = aiRaiseTarget - aiPlayer.bet;
                
                // 确保加注不超过剩余筹码
                const actualRaiseAmount = Math.min(additionalBet, aiPlayer.chips);
                
                // 更新玩家筹码和下注
                aiPlayer.chips -= actualRaiseAmount;
                aiPlayer.bet += actualRaiseAmount;
                
                // 更新当前下注 - 保证它是加注后的实际下注金额
                gameState.currentBet = aiPlayer.bet;
                console.log(`AI玩家 ${aiPlayer.name} 加注 ${actualRaiseAmount}，当前下注更新为 ${gameState.currentBet}`);
                
                // 更新底池
                gameState.pot += actualRaiseAmount;
                        
                        // 添加历史记录
                addHistoryItem(aiPlayer.name, 'raise', actualRaiseAmount);
                        
                // 显示加注对话
                showPlayerDialog(aiPlayer.id, 'raise');
                
                // 创建筹码动画
                        createChipAnimation(
                    PLAYER_POSITIONS[aiPlayer.id],
                    { top: '50%', left: '50%' },
                    actualRaiseAmount
                );
                
                // 播放加注音效
                playSound('raise');
                
                break;
                
            case 'allIn':
                // 计算全押金额
                const allInAmount = aiPlayer.chips;
                
                // 更新下注和底池
                aiPlayer.bet += allInAmount;
                aiPlayer.chips = 0;
                        gameState.pot += allInAmount;
                
                // 如果全押金额超过当前最大下注，更新当前下注
                if (aiPlayer.bet > gameState.currentBet) {
                    gameState.currentBet = aiPlayer.bet;
                }
                
                // 标记为全押
                aiPlayer.isAllIn = true;
                        
                        // 添加历史记录
                addHistoryItem(aiPlayer.name, 'allIn', allInAmount);
                        
                // 显示全押对话
                showPlayerDialog(aiPlayer.id, 'allIn');
                
                // 创建筹码动画
                        createChipAnimation(
                    PLAYER_POSITIONS[aiPlayer.id],
                    { top: '50%', left: '50%' },
                            allInAmount
                        );
                
                // 播放全押音效
                playSound('raise');
                
                    break;
            }
                
                // 更新显示
                updateGameDisplay();
                
        // 进入下一个玩家
                console.log("AI行动完成，准备进入下一个玩家");
                setTimeout(() => {
                    console.log("调用afterPlayerAction处理下一步");
                    afterPlayerAction();
                }, 1500);
    }, 1000 + Math.random() * 1000); // 1-2秒的随机思考时间
}

// 显示思考动画
function displayThinkingAnimation(playerId) {
    // 显示思考对话
    showPlayerDialog(playerId, 'thinking');
}

// 显示玩家对话
function showPlayerDialog(playerId, type) {
    // 检查玩家ID是否有效
    if (playerId < 0 || playerId >= gameState.players.length) {
        console.error(`无效的玩家ID: ${playerId}`);
        return;
    }
    
    const player = gameState.players[playerId];
    if (!player) {
        console.error(`未找到ID为 ${playerId} 的玩家`);
        return;
    }
    
    console.log(`显示玩家 ${player.name} 的 ${type} 对话`);
    const dialogText = getRandomDialog(type, player.name);
    createDialogBubble(dialogText, PLAYER_POSITIONS[playerId], playerId);
}

// 玩家完成行动后的处理
function afterPlayerAction() {
    console.log("-------------------");
    console.log("进入afterPlayerAction函数");
    
    // 检查是否所有玩家都已行动或弃牌
    let allPlayersActed = true;
    let highestBet = 0;
    
    // 找出最高下注额
    gameState.players.forEach(player => {
        if (player.bet > highestBet) {
            highestBet = player.bet;
        }
    });
    
    console.log(`当前最高下注: ${highestBet}`);
    
    // 检查是否所有未弃牌的玩家都已经行动并且下注金额相等或已经全押
    for (let i = 0; i < gameState.players.length; i++) {
        const player = gameState.players[i];
        
        // 跳过已弃牌或全押或没筹码的玩家
        if (player.folded || player.isAllIn || player.chips === 0) {
            console.log(`玩家${player.name}已弃牌/全押/没筹码，跳过检查`);
            continue;
        }
        
        // 如果有玩家下注额不等于最高下注，且有能力跟注，表示还没完成行动
        if (player.bet < highestBet && player.chips > 0) {
            allPlayersActed = false;
            console.log(`玩家${player.name}下注(${player.bet})小于最高下注(${highestBet})且还有筹码(${player.chips})，需要继续行动`);
            break;
        }
    }
    
    // 计算还有多少活跃玩家
    const activePlayers = gameState.players.filter(p => !p.folded);
    console.log(`活跃玩家数量: ${activePlayers.length}`);
    
    // 如果只剩一个未弃牌的玩家，直接进入摊牌阶段
    if (activePlayers.length === 1) {
        console.log("只剩一个玩家未弃牌，直接进入摊牌阶段");
        // 快速进入摊牌阶段
        gameState.gamePhase = 'river';
        advanceGamePhase();
        return;
    }
    
    // 如果所有玩家都已经行动，进入下一阶段
    if (allPlayersActed) {
        console.log("所有玩家已完成行动，进入下一阶段");
        advanceGamePhase();
        return;
    }
    
    // 否则，继续到下一个玩家
    console.log("还有玩家需要行动，移动到下一个玩家");
    moveToNextPlayer();
}

// 移动到下一个玩家
function moveToNextPlayer() {
    console.log("执行moveToNextPlayer");
    
    // 清除当前计时器
    clearActionTimer();
    
    // 如果当前玩家索引无效，重置为0
    if (gameState.currentPlayer < 0 || gameState.currentPlayer >= gameState.players.length) {
        console.error("当前玩家索引无效:", gameState.currentPlayer);
        gameState.currentPlayer = 0;
    }
    
    // 找到下一个未弃牌且未全押的玩家
    let nextPlayerFound = false;
    let startIndex = (gameState.currentPlayer + 1) % gameState.players.length;
    let checkedPlayers = 0; // 防止无限循环
    
    while (checkedPlayers < gameState.players.length) {
        const nextIndex = (startIndex + checkedPlayers) % gameState.players.length;
        const nextPlayer = gameState.players[nextIndex];
        
        // 检查这个玩家是否可以行动
        if (!nextPlayer.folded && !nextPlayer.isAllIn && nextPlayer.chips > 0) {
            gameState.currentPlayer = nextIndex;
            nextPlayerFound = true;
            console.log(`找到下一个玩家: ${nextPlayer.name}, 索引: ${nextIndex}`);
            break;
        }
        
        checkedPlayers++;
    }
    
    // 如果没找到下一个有效玩家，进入下一阶段
    if (!nextPlayerFound) {
        console.log("没找到可行动的玩家，进入下一阶段");
        advanceGamePhase();
        return;
    }
    
    // 更新当前玩家显示
    updateCurrentPlayerDisplay();
    
    // 更新历史记录中的当前行动玩家
    updateCurrentActionPlayer(gameState.currentPlayer);
    
    // 如果当前玩家是主玩家
    if (gameState.currentPlayer === 0) {
        // 思考气泡
        const mainPlayer = gameState.players[0];
        const thoughtText = getRandomDialog('thinking');
        createDialogBubble(thoughtText, PLAYER_POSITIONS[0], 0);
        
        // 启用控制按钮
        foldBtn.disabled = false;
        checkBtn.disabled = false;
        callBtn.disabled = false;
        raiseBtn.disabled = false;
        allInBtn.disabled = false;
        
        // 更新按钮状态
        updateButtonStates();
        
        // 设置行动计时器
        setTimeout(() => {
        startActionTimer();
        }, 500);
    } else {
        // AI玩家行动
        // 先启动计时器，再执行AI动作
        setTimeout(() => {
            startActionTimer();
        simulateAIActions();
        }, 500);
    }
}

// 重新开始游戏
function handleRestart() {
    // 清除计时器
    clearActionTimer();
    
    // 确认是否要重新开始
    if (confirm('确定要重新开始游戏吗？您将失去当前游戏的所有进度。')) {
        // 清除所有对话气泡和动画
        dialogContainer.innerHTML = '';
        animationContainer.innerHTML = '';
        
        // 清除所有玩家的手牌显示
        document.querySelectorAll('.player-cards-display').forEach(el => {
            el.remove();
        });
        
        // 禁用游戏按钮
        foldBtn.disabled = true;
        checkBtn.disabled = true;
        callBtn.disabled = true;
        raiseBtn.disabled = true;
        allInBtn.disabled = true;
        
        // 重置游戏状态
        resetGameState();
        
        // 清空玩家容器
        playersContainer.innerHTML = '';
        
        // 清空公共牌和玩家手牌
        communityCardElements.forEach(card => {
            card.textContent = '';
            card.className = 'card-placeholder';
            card.style.backgroundColor = 'rgba(255, 255, 255, 0.1)';
        });
        
        playerCardElements.forEach(card => {
            card.textContent = '';
            card.className = 'card-placeholder';
            card.style.backgroundColor = 'rgba(255, 255, 255, 0.1)';
        });
        
        // 更新显示
        potDisplay.textContent = '底池: $0';
        potChipsDisplay.innerHTML = '';
        gamePhaseDisplay.textContent = PHASE_DESCRIPTIONS.waiting;
        currentTurnDisplay.textContent = '轮到: -';
        
        // 清空玩家容器
        playersContainer.innerHTML = '';
        
        // 重新开始游戏
        startGame(gameState.players[0]?.name || playerNameInput.value.trim() || "玩家1");
    }
}

// 添加历史记录
function addHistoryItem(playerName, actionType, amount = null, isPhaseChange = false, isCurrent = false) {
    // 创建历史记录项
    const historyItem = document.createElement('div');
    historyItem.className = 'history-item';
    
    if (isPhaseChange) {
        historyItem.classList.add('phase-change');
        historyItem.textContent = actionType; // 阶段变更直接显示文本
    } else {
        if (isCurrent) {
            historyItem.classList.add('current-action');
        }
        
        // 创建玩家行动内容
        const playerAction = document.createElement('div');
        playerAction.className = 'player-action';
        
        // 玩家名称
        const nameSpan = document.createElement('span');
        nameSpan.className = 'player-name';
        nameSpan.textContent = playerName;
        
        // 行动类型
        const actionSpan = document.createElement('span');
        actionSpan.className = 'action-type';
        actionSpan.textContent = ACTION_DESCRIPTIONS[actionType] || actionType;
        
        // 添加到玩家行动
        playerAction.appendChild(nameSpan);
        playerAction.appendChild(document.createTextNode('：'));
        playerAction.appendChild(actionSpan);
        
        // 如果有金额，添加金额
        if (amount !== null) {
            const amountSpan = document.createElement('span');
            amountSpan.className = 'action-amount';
            amountSpan.textContent = ` $${amount}`;
            playerAction.appendChild(amountSpan);
        }
        
        // 添加到历史项
        historyItem.appendChild(playerAction);
    }
    
    // 添加到历史记录容器
    historyContent.appendChild(historyItem);
    
    // 滚动到底部
    historyContent.scrollTop = historyContent.scrollHeight;
    
    // 保存到游戏状态
    gameState.history.push({
        playerName,
        actionType,
        amount,
        isPhaseChange,
        timestamp: new Date().getTime()
    });
}

// 清空历史记录
function clearHistory() {
    historyContent.innerHTML = '';
    gameState.history = [];
}

// 更新当前行动玩家
function updateCurrentActionPlayer(playerIndex) {
    // 移除所有current-action类
    const currentItems = document.querySelectorAll('.history-item.current-action');
    currentItems.forEach(item => {
        item.classList.remove('current-action');
    });
    
    // 如果有效索引，添加新的当前行动指示
    if (playerIndex >= 0 && playerIndex < gameState.players.length) {
        const player = gameState.players[playerIndex];
        
        // 添加一个临时的"轮到谁"记录
        const lastItem = historyContent.lastChild;
        if (lastItem && lastItem.textContent.includes('轮到')) {
            historyContent.removeChild(lastItem);
        }
        
        const historyItem = document.createElement('div');
        historyItem.className = 'history-item current-action';
        historyItem.textContent = `轮到 ${player.name} 行动`;
        historyContent.appendChild(historyItem);
        
        // 滚动到底部
        historyContent.scrollTop = historyContent.scrollHeight;
    }
}

// 推进游戏阶段
function advanceGamePhase() {
    switch (gameState.gamePhase) {
        case 'preflop':
            dealFlop();
            gameState.gamePhase = 'flop';
            break;
        case 'flop':
            dealTurn();
            gameState.gamePhase = 'turn';
            break;
        case 'turn':
            dealRiver();
            gameState.gamePhase = 'river';
            break;
        case 'river':
            showdown();
            gameState.gamePhase = 'showdown';
            break;
        case 'showdown':
            setTimeout(startNewRound, 3000);
            break;
    }
    
    // 更新游戏阶段显示
    updateGamePhaseDisplay();
    
    // 如果不是摊牌阶段，开始新的下注回合
    if (gameState.gamePhase !== 'showdown') {
        setTimeout(() => {
            startBettingRound();
        }, 1500);
    }
    
    // 添加阶段变更记录
    addHistoryItem('系统', `进入${PHASE_DESCRIPTIONS[gameState.gamePhase]}`, null, true);
}

// 发翻牌
function dealFlop() {
    // 烧掉一张牌
    gameState.deck.pop();
    
    // 发3张公共牌
    for (let i = 0; i < 3; i++) {
        setTimeout(() => {
            gameState.communityCards.push(gameState.deck.pop());
            renderCommunityCards();
        }, 300 * i);
    }
}

// 发转牌
function dealTurn() {
    // 烧掉一张牌
    gameState.deck.pop();
    
    // 发第4张公共牌
    setTimeout(() => {
        gameState.communityCards.push(gameState.deck.pop());
        renderCommunityCards();
    }, 300);
}

// 发河牌
function dealRiver() {
    // 烧掉一张牌
    gameState.deck.pop();
    
    // 发第5张公共牌
    setTimeout(() => {
        gameState.communityCards.push(gameState.deck.pop());
        renderCommunityCards();
    }, 300);
}

// 摊牌
function showdown() {
    console.log("开始摊牌阶段");
    
    // 收集所有未弃牌的玩家
    const activePlayers = gameState.players.filter(player => !player.folded);
    
    // 显示所有活跃玩家的手牌
    setTimeout(() => {
        showAllPlayersCards(activePlayers);
    }, 500);
    
    // 如果只有一个玩家未弃牌，直接宣布胜利
    if (activePlayers.length === 1) {
        const winner = activePlayers[0];
        console.log(`只有一名未弃牌玩家 ${winner.name}，直接获胜`);
        
        // 更新玩家筹码
        winner.chips += gameState.pot;
        
        // 添加赢家记录
        addHistoryItem(winner.name, '赢得底池', gameState.pot, false);
        
        declareWinner(winner);
        return;
    }
    
    // 否则，使用简化的随机选择赢家方法
        const winnerIndex = Math.floor(Math.random() * activePlayers.length);
        const winner = activePlayers[winnerIndex];
        winner.chips += gameState.pot;
        
        // 添加赢家记录
        addHistoryItem(winner.name, '赢得底池', gameState.pot, false);
        
        // 提示赢家
        setTimeout(() => {
        declareWinner(winner);
    }, 1000);
    
    // 检查锦标赛是否结束
    checkTournamentEnd();
    
    // 辅助函数：宣布胜利者
    function declareWinner(winner, isSplit = false) {
            // 播放获胜音效
            playSound('win');
        
        // 创建获胜通知
        const winMessage = isSplit ? 
            `${winner.name} 等人平分了底池 $${gameState.pot}!` : 
            `${winner.name} 赢得了 $${gameState.pot}!`;
        
        createActionNotification(winMessage, { left: '50%', top: '40%' });
            
            // 高亮获胜玩家
            if (winner.id === 0) {
                // 主玩家获胜
                // 筹码动画 - 从中心到玩家
                createChipAnimation(
                    { left: '50%', top: '50%' },
                    { left: '50%', top: '85%' },
                    gameState.pot
                );
                
                // 获胜对话
            const winText = getRandomDialog('win', winner.name);
                createDialogBubble(winText, PLAYER_POSITIONS[0], 0);
            } else {
                // AI获胜
                const winnerElement = document.getElementById(`player${winner.id}`);
                if (winnerElement) {
                    winnerElement.classList.add('winner');
                    
                    // 筹码动画 - 从中心到获胜玩家
                    createChipAnimation(
                        { left: '50%', top: '50%' },
                        {
                            left: PLAYER_POSITIONS[winner.id].left,
                            top: PLAYER_POSITIONS[winner.id].top
                        },
                        gameState.pot
                    );
                    
                    // 获胜对话
                const winText = getRandomDialog('win', winner.name);
                    createDialogBubble(
                        winText, 
                        {
                            left: PLAYER_POSITIONS[winner.id].left,
                            top: PLAYER_POSITIONS[winner.id].top
                        }, 
                        winner.id
                    );
                    
                    // 移除获胜动画
                    setTimeout(() => {
                        winnerElement.classList.remove('winner');
                    }, 3000);
                }
            }
            
            // 重置所有下注
            gameState.players.forEach(player => {
                player.bet = 0;
            });
            
            gameState.pot = 0;
            updateGameDisplay();
            
            // 锦标赛模式 - 检查是否有玩家被淘汰
            if (gameState.tournament.isEnabled) {
                // 检查所有玩家的筹码，找出被淘汰的玩家
                gameState.players.forEach((player, index) => {
                    if (player.chips <= 0 && !player.eliminated) {
                        player.eliminated = true;
                        handlePlayerElimination(index);
                    }
                });
                
                // 更新锦标赛显示
                updateTournamentDisplay();
            }
            
            // 如果锦标赛已结束，不继续游戏
            if (gameState.tournament.isEnabled && gameState.tournament.isFinished) {
                return;
            }
    }
}

// 检查锦标赛是否结束
function checkTournamentEnd() {
    // 计算剩余玩家数量
    const activePlayers = gameState.players.filter(player => player.chips > 0);
    
    // 如果只剩一个玩家，锦标赛结束
    if (activePlayers.length === 1) {
        // 将剩余玩家标记为冠军
        const champion = activePlayers[0];
        
        // 设置锦标赛结束状态
        gameState.tournament.isFinished = true;
        
        // 清除级别计时器
        if (gameState.tournament.levelTimer) {
            clearInterval(gameState.tournament.levelTimer);
            gameState.tournament.levelTimer = null;
        }
        
        // 添加锦标赛结束记录
        addHistoryItem('系统', `锦标赛结束! ${champion.name} 获得冠军!`, null, true);
        
        // 延迟显示结果界面，让用户看到最后一局
        setTimeout(() => {
            showTournamentResults(champion);
        }, 3000);
    }
}

// 显示锦标赛结果
function showTournamentResults(champion) {
    // 显示结果界面
    tournamentResults.style.display = 'flex';
    
    // 播放胜利音效
    playSound('win');
    
    // 设置冠军名称并添加打字效果
    tournamentWinnerDisplay.textContent = '';
    let charIndex = 0;
    const typeWinnerText = () => {
        if (charIndex < champion.name.length) {
            tournamentWinnerDisplay.textContent += champion.name.charAt(charIndex);
            charIndex++;
            setTimeout(typeWinnerText, 100);
        } else {
            // 打字效果结束后添加闪烁效果
            tournamentWinnerDisplay.classList.add('text-glow');
        }
    };
    typeWinnerText();
    
    // 清空结果表
    resultsTableBody.innerHTML = '';
    
    // 计算奖金
    const finalResults = [];
    
    // 首先添加冠军
    finalResults.push({
        position: 1,
        name: champion.name,
        prize: Math.floor(gameState.tournament.prizePool * TOURNAMENT_PAYOUTS[0].percentage / 100)
    });
    
    // 然后按照淘汰顺序逆序添加其他玩家
    const reversedEliminations = [...gameState.tournament.eliminatedPlayers].reverse();
    
    for (let i = 0; i < reversedEliminations.length; i++) {
        const player = reversedEliminations[i];
        const position = i + 2; // 位置从2开始（1是冠军）
        
        let prize = 0;
        // 找到对应的奖金比例
        const payoutEntry = TOURNAMENT_PAYOUTS.find(p => p.position === position);
        if (payoutEntry) {
            prize = Math.floor(gameState.tournament.prizePool * payoutEntry.percentage / 100);
        }
        
        finalResults.push({
            position: position,
            name: player.name,
            prize: prize
        });
    }
    
    // 添加动画效果逐行显示结果
    finalResults.forEach((result, index) => {
        setTimeout(() => {
            // 创建新行
            const row = document.createElement('tr');
            
            // 根据名次添加特殊样式
            if (result.position === 1) {
                row.className = 'first-place';
            } else if (result.position === 2) {
                row.className = 'second-place';
            } else if (result.position === 3) {
                row.className = 'third-place';
            }
            
            // 添加名次单元格
            const posCell = document.createElement('td');
            
            // 为前三名添加特殊图标
            if (result.position === 1) {
                posCell.innerHTML = `<span class="medal gold">🥇 1</span>`;
            } else if (result.position === 2) {
                posCell.innerHTML = `<span class="medal silver">🥈 2</span>`;
            } else if (result.position === 3) {
                posCell.innerHTML = `<span class="medal bronze">🥉 3</span>`;
            } else {
                posCell.textContent = result.position;
            }
            
            // 添加玩家名称单元格
            const nameCell = document.createElement('td');
            nameCell.textContent = result.name;
            
            // 添加奖金单元格
            const prizeCell = document.createElement('td');
            
            // 为有奖金的玩家添加动画显示奖金
            if (result.prize > 0) {
                let displayPrize = 0;
                const increment = Math.ceil(result.prize / 20); // 20步显示完整奖金
                const updatePrize = () => {
                    displayPrize = Math.min(displayPrize + increment, result.prize);
                    prizeCell.textContent = `$${displayPrize}`;
                    
                    if (displayPrize < result.prize) {
                        // 继续更新
                        setTimeout(updatePrize, 50);
                    } else {
                        // 最终奖金显示后添加闪光效果
                        prizeCell.classList.add('prize-glow');
                    }
                };
                updatePrize();
            } else {
                prizeCell.textContent = '-';
            }
            
            // 将单元格添加到行
            row.appendChild(posCell);
            row.appendChild(nameCell);
            row.appendChild(prizeCell);
            
            // 将行添加到表格并设置动画
            row.style.opacity = '0';
            resultsTableBody.appendChild(row);
            
            // 淡入效果
    setTimeout(() => {
                row.style.transition = 'opacity 0.5s ease';
                row.style.opacity = '1';
            }, 50);
            
            // 播放添加行时的音效
            if (index === 0) {
                playSound('win'); // 冠军行使用win音效
            } else {
                playSound('chipStack'); // 其他行使用筹码音效
            }
        }, index * 500); // 每500毫秒添加一行
    });
    
    // 注册返回主菜单按钮事件
    const backToLobbyBtn = document.getElementById('backToLobbyBtn');
    if (backToLobbyBtn) {
        backToLobbyBtn.addEventListener('click', () => {
            // 隐藏结果界面
            tournamentResults.style.display = 'none';
            // 重置游戏状态
            resetGameState();
            // 显示欢迎界面
            welcomeScreen.style.display = 'flex';
            gameScreen.style.display = 'none';
        });
    }
    
    // 添加全局烟花效果
    createFireworks();
}

// 创建烟花特效
function createFireworks() {
    // 创建烟花容器
    const fireworksContainer = document.createElement('div');
    fireworksContainer.className = 'fireworks-container';
    document.body.appendChild(fireworksContainer);
    
    // 发射多个烟花
    for (let i = 0; i < 8; i++) {
        setTimeout(() => {
            const firework = document.createElement('div');
            firework.className = 'firework';
            
            // 随机位置
            const randomLeft = Math.random() * 90 + 5; // 5-95%
            const randomBottom = Math.random() * 40 + 30; // 30-70%
            
            firework.style.left = `${randomLeft}%`;
            firework.style.bottom = `${randomBottom}%`;
            
            // 随机颜色
            const colors = ['#ffbe0b', '#fb5607', '#ff006e', '#8338ec', '#3a86ff'];
            const randomColor = colors[Math.floor(Math.random() * colors.length)];
            firework.style.setProperty('--firework-color', randomColor);
            
            // 随机大小
            const size = Math.random() * 5 + 5; // 5-10vh
            firework.style.setProperty('--firework-size', `${size}vh`);
            
            fireworksContainer.appendChild(firework);
            
            // 发射效果
    setTimeout(() => {
                // 爆炸效果
                firework.classList.add('explode');
                
                // 爆炸后移除
                setTimeout(() => {
                    firework.remove();
                    
                    // 最后一个烟花后清除容器
                    if (i === 7 && !fireworksContainer.hasChildNodes()) {
                        fireworksContainer.remove();
                    }
    }, 1500);
            }, Math.random() * 200 + 100);
        }, i * 600); // 每600毫秒发射一个烟花
    }
}

// 添加位置因素影响函数
function evaluatePosition(position) {
    // 晚位置加成，早位置减弱
    if (position === 'late') return 0.15;
    if (position === 'middle') return 0.05;
    if (position === 'early') return -0.05;
    if (position === 'blinds') return -0.1;
    return 0;
}

// 根据庄家位置计算相对位置
function getRelativePositions(dealerPosition, playerCount) {
    const positions = [];
    for (let i = 0; i < playerCount; i++) {
        // 计算相对位置
        const relativePosition = (i - dealerPosition - 3 + playerCount) % playerCount;
        const positionPercentile = relativePosition / playerCount;
        
        // 根据位置百分比分配位置名称
        let positionName;
        if (i === (dealerPosition + 1) % playerCount || i === (dealerPosition + 2) % playerCount) {
            positionName = 'blinds'; // 小盲和大盲
        } else if (positionPercentile < 0.3) {
            positionName = 'early'; // 早位
        } else if (positionPercentile < 0.7) {
            positionName = 'middle'; // 中位
        } else {
            positionName = 'late'; // 晚位
        }
        positions[i] = positionName;
    }
    return positions;
}
// AI决策逻辑
function makeAIDecision(playerIndex) {
    // 计算手牌强度
    const player = gameState.players[playerIndex];
    let handStrength = calculateHandStrength(player.hand, gameState.communityCards);
    
    // 增加位置因素
    const positions = getRelativePositions(gameState.dealerPosition, gameState.players.length);
    const positionFactor = evaluatePosition(positions[playerIndex]);
    handStrength *= (1 + positionFactor);
    
    // 继续使用第一个makeAIDecision函数的逻辑
    return gameState.players[playerIndex].folded ? 'fold' : 'check';
}

// 计算手牌强度 (用于AI决策)
function calculateHandStrength(hand, communityCards, activePlayers) {
    // 如果还没有手牌，返回中等强度
    if (!hand || hand.length === 0) return 0.5;
    
    // 根据游戏阶段不同，计算策略不同
    const allCards = [...hand, ...communityCards];
    
    // 前翻牌圈 (只有手牌) - 根据起手牌价值评估
    if (communityCards.length === 0) {
        return evaluatePreFlopHand(hand, activePlayers);
    }
    
    // 其他轮次 - 基于实际牌力和胜率估计
    return evaluatePostFlopStrength(hand, communityCards, activePlayers);
}

// 评估前翻牌圈手牌强度
function evaluatePreFlopHand(hand, activePlayers) {
    if (hand.length !== 2) return 0.5;
    
    const card1 = hand[0];
    const card2 = hand[1];
    
    // 提取牌值和花色
    const value1 = card1.value;
    const value2 = card2.value;
    const suit1 = card1.suit;
    const suit2 = card2.suit;
    
    // 转换牌值到强度数值
    const valueStrength = {
        '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
        '9': 9, '10': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14
    };
    
    const v1 = valueStrength[value1];
    const v2 = valueStrength[value2];
    
    // 是否同花
    const suited = suit1 === suit2;
    
    // 是否对子
    const paired = v1 === v2;
    
    // 牌值接近度 (连牌可能性)
    const valueDiff = Math.abs(v1 - v2);
    const connected = valueDiff <= 3;
    
    let strength = 0;
    
    // 1. 高对子
    if (paired) {
        if (v1 >= 10) { // AA, KK, QQ, JJ, TT
            strength = 0.85 - (14 - v1) * 0.03;
        } else { // 小对子
            strength = 0.65 - (10 - v1) * 0.04;
        }
    }
    // 2. 高牌同花
    else if (suited) {
        if (v1 >= 12 && v2 >= 10) { // AK, AQ, KQ同花
            strength = 0.75;
        } else if (v1 >= 10 && v2 >= 8) { // 其他高牌同花
            strength = 0.68;
        } else if (connected) { // 同花连牌
            strength = 0.65 - valueDiff * 0.03;
        } else { // 普通同花
            strength = 0.55;
        }
    }
    // 3. 非同花高牌
    else if (v1 >= 12 && v2 >= 10) { // AK, AQ, KQ
        strength = 0.7;
    }
    // 4. 连牌
    else if (connected) {
        if (v1 >= 10 && v2 >= 8) { // 高连牌
            strength = 0.62;
        } else { // 其他连牌
            strength = 0.5 - valueDiff * 0.03;
        }
    }
    // 5. 其他牌型
    else {
        const highCard = Math.max(v1, v2);
        const lowCard = Math.min(v1, v2);
        
        if (highCard === 14) { // 含A的牌
            strength = 0.5 - (14 - lowCard) * 0.02;
        } else if (highCard >= 11) { // 含K,Q,J的牌
            strength = 0.4 - (11 - lowCard) * 0.01;
        } else { // 低价值牌
            strength = 0.3;
        }
    }
    
    // 多人游戏时，起手牌价值下降(玩家越多，成牌难度越大)
    strength -= (activePlayers - 2) * 0.02;
    
    // 返回0-1之间的强度值
    return Math.max(0.1, Math.min(strength, 0.9));
}

// 评估翻牌后的手牌强度
function evaluatePostFlopStrength(hand, communityCards, activePlayers) {
    if (hand.length + communityCards.length < 5) {
        // 如果牌不够5张，计算当前最佳可能牌型
        const currentBest = evaluateHandRank([...hand, ...communityCards]);
        return adjustStrengthByRank(currentBest, communityCards.length, activePlayers);
    }
    
    // 计算当前最佳牌型
    const allCards = [...hand, ...communityCards];
    const bestRank = evaluateHandRank(allCards);
    
    return adjustStrengthByRank(bestRank, communityCards.length, activePlayers);
}

// 根据牌型和游戏阶段调整强度
function adjustStrengthByRank(rank, communityCardCount, activePlayers) {
    // 基础强度值映射
    const baseStrength = {
        'high_card': 0.1,
        'pair': 0.3,
        'two_pair': 0.5,
        'three_of_a_kind': 0.6,
        'straight': 0.7,
        'flush': 0.75,
        'full_house': 0.85,
        'four_of_a_kind': 0.95,
        'straight_flush': 0.98,
        'royal_flush': 1.0
    };
    
    let strength = baseStrength[rank.type] || 0.5;
    
    // 根据公共牌数量调整
    // 翻牌圈时，估值应该更保守
    if (communityCardCount === 3) {
        strength *= 0.85;
    } 
    // 转牌圈，可以更确定一些
    else if (communityCardCount === 4) {
        strength *= 0.95;
    }
    
    // 多人游戏时需要更强的牌
    strength -= (activePlayers - 2) * 0.03;
    
    return Math.max(0.1, Math.min(strength, 0.98));
}

// 评估牌型
function evaluateHandRank(cards) {
    // 如果牌不足5张，返回当前最高牌型
    if (cards.length < 5) {
        // 简化评估，仅检查对子、三条等
        return evaluateIncompletHand(cards);
    }
    
    // 转换牌值格式以便评估
    const valueMap = {
        '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
        '9': 9, '10': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14
    };
    
    // 统计每个牌值和花色的数量
    const valueCounts = {};
    const suitCounts = {};
    const cardsByValue = [];
    
    for (const card of cards) {
        const value = valueMap[card.value];
        const suit = card.suit;
        
        valueCounts[value] = (valueCounts[value] || 0) + 1;
        suitCounts[suit] = (suitCounts[suit] || 0) + 1;
        
        cardsByValue.push({
            value: value,
            suit: suit
        });
    }
    
    // 按牌值排序 (从大到小)
    cardsByValue.sort((a, b) => b.value - a.value);
    
    // 检查是否同花
    let isFlush = false;
    let flushSuit = null;
    for (const suit in suitCounts) {
        if (suitCounts[suit] >= 5) {
            isFlush = true;
            flushSuit = suit;
            break;
        }
    }
    
    // 检查是否顺子
    let isStraight = false;
    let straightHighCard = null;
    
    // 获取不重复的牌值数组
    const uniqueValues = Object.keys(valueCounts).map(Number).sort((a, b) => b - a);
    
    // 特殊情况: A-5-4-3-2 顺子
    if (uniqueValues.includes(14) && uniqueValues.includes(5) && 
        uniqueValues.includes(4) && uniqueValues.includes(3) && uniqueValues.includes(2)) {
        isStraight = true;
        straightHighCard = 5; // 5高顺子
    } else {
        // 检查常规顺子
        for (let i = 0; i < uniqueValues.length - 4; i++) {
            if (uniqueValues[i] - uniqueValues[i + 4] === 4) {
                isStraight = true;
                straightHighCard = uniqueValues[i];
                break;
            }
        }
    }
    
    // 同花顺 & 皇家同花顺
    if (isFlush && isStraight) {
        // 检查是否同花顺
        const flushCards = cardsByValue.filter(card => card.suit === flushSuit);
        let isFlushStraight = false;
        let flushStraightHighCard = null;
        
        // 获取同花的不重复牌值
        const uniqueFlushValues = [...new Set(flushCards.map(card => card.value))].sort((a, b) => b - a);
        
        // 特殊情况: 同花A-5-4-3-2顺子
        if (uniqueFlushValues.includes(14) && uniqueFlushValues.includes(5) && 
            uniqueFlushValues.includes(4) && uniqueFlushValues.includes(3) && uniqueFlushValues.includes(2)) {
            isFlushStraight = true;
            flushStraightHighCard = 5;
    } else {
            // 检查常规同花顺
            for (let i = 0; i < uniqueFlushValues.length - 4; i++) {
                if (uniqueFlushValues[i] - uniqueFlushValues[i + 4] === 4) {
                    isFlushStraight = true;
                    flushStraightHighCard = uniqueFlushValues[i];
                    break;
                }
            }
        }
        
        // 确认是同花顺
        if (isFlushStraight) {
            // 皇家同花顺 (10 J Q K A)
            if (flushStraightHighCard === 14) {
                return { type: 'royal_flush', value: 14 };
            }
            // 普通同花顺
            return { type: 'straight_flush', value: flushStraightHighCard };
        }
    }
    
    // 检查四条
    for (const value in valueCounts) {
        if (valueCounts[value] === 4) {
            return { type: 'four_of_a_kind', value: Number(value) };
        }
    }
    
    // 检查葫芦 (三条+对子)
    let hasThreeOfKind = false;
    let threeOfKindValue = null;
    let hasPair = false;
    let pairValue = null;
    
    for (const value in valueCounts) {
        if (valueCounts[value] === 3) {
            if (!hasThreeOfKind || Number(value) > threeOfKindValue) {
                hasThreeOfKind = true;
                threeOfKindValue = Number(value);
            }
        } else if (valueCounts[value] === 2) {
            hasPair = true;
            if (!pairValue || Number(value) > pairValue) {
                pairValue = Number(value);
            }
        }
    }
    
    if (hasThreeOfKind && hasPair) {
        return { type: 'full_house', value: threeOfKindValue, kicker: pairValue };
    }
    
    // 检查同花
    if (isFlush) {
        const flushCards = cardsByValue.filter(card => card.suit === flushSuit);
        return { type: 'flush', values: flushCards.slice(0, 5).map(c => c.value) };
    }
    
    // 检查顺子
    if (isStraight) {
        return { type: 'straight', value: straightHighCard };
    }
    
    // 检查三条
    if (hasThreeOfKind) {
        return { type: 'three_of_a_kind', value: threeOfKindValue };
    }
    
    // 检查两对
    const pairs = [];
    for (const value in valueCounts) {
        if (valueCounts[value] === 2) {
            pairs.push(Number(value));
        }
    }
    
    if (pairs.length >= 2) {
        pairs.sort((a, b) => b - a);
        return { type: 'two_pair', values: [pairs[0], pairs[1]] };
    }
    
    // 检查对子
    if (hasPair) {
        return { type: 'pair', value: pairValue };
    }
    
    // 高牌
    return { type: 'high_card', value: cardsByValue[0].value };
}

// 评估不完整的手牌 (少于5张)
function evaluateIncompletHand(cards) {
    // 转换牌值格式
    const valueMap = {
        '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
        '9': 9, '10': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14
    };
    
    // 统计每个牌值和花色
    const valueCounts = {};
    const suitCounts = {};
    
    for (const card of cards) {
        const value = valueMap[card.value];
        const suit = card.suit;
        
        valueCounts[value] = (valueCounts[value] || 0) + 1;
        suitCounts[suit] = (suitCounts[suit] || 0) + 1;
    }
    
    // 检查三条
    for (const value in valueCounts) {
        if (valueCounts[value] === 3) {
            return { type: 'three_of_a_kind', value: Number(value) };
        }
    }
    
    // 检查对子
    const pairs = [];
    for (const value in valueCounts) {
        if (valueCounts[value] === 2) {
            pairs.push(Number(value));
        }
    }
    
    if (pairs.length >= 2) {
        pairs.sort((a, b) => b - a);
        return { type: 'two_pair', values: pairs.slice(0, 2) };
    }
    
    if (pairs.length === 1) {
        return { type: 'pair', value: pairs[0] };
    }
    
    // 检查是否可能同花
    for (const suit in suitCounts) {
        if (suitCounts[suit] === cards.length) {
            return { type: 'high_card', value: Math.max(...Object.keys(valueCounts).map(Number)), flushDraw: true };
        }
    }
    
    // 检查是否可能顺子
    const values = Object.keys(valueCounts).map(Number).sort((a, b) => a - b);
    if (values.length >= 2) {
        if (values[values.length - 1] - values[0] <= 4) {
            return { type: 'high_card', value: Math.max(...values), straightDraw: true };
        }
    }
    
    // 高牌
    return { type: 'high_card', value: Math.max(...Object.keys(valueCounts).map(Number)) };
}

// 初始化游戏
document.addEventListener('DOMContentLoaded', initGame); 

// 初始化锦标赛模式
function initializeTournament() {
    // 设置初始盲注
    gameState.smallBlind = 5;
    gameState.bigBlind = 10;
    
    // 更新盲注显示
    document.getElementById('currentBlinds').textContent = `${gameState.smallBlind}/${gameState.bigBlind}`;
    
    // 设置当前级别
    gameState.tournament.currentLevel = 0;
    
    // 更新级别显示
    document.getElementById('currentLevel').textContent = gameState.tournament.currentLevel + 1;
    
    // 更新剩余玩家显示
    const remainingPlayers = gameState.players.filter(p => !p.eliminated).length;
    document.getElementById('playersRemaining').textContent = remainingPlayers;
    
    // 设置级别计时器
    startLevelTimer();
    
    // 添加锦标赛开始记录
    addHistoryItem('系统', '锦标赛模式开始', null, true);
    addHistoryItem('系统', `初始盲注: ${gameState.smallBlind}/${gameState.bigBlind}`, null, true);
}

// 启动级别计时器
function startLevelTimer() {
    // 如果已有计时器，先清除
    if (gameState.tournament.levelTimer) {
        clearInterval(gameState.tournament.levelTimer);
    }
    
    // 设置级别时间（分钟转换为秒）
    gameState.tournament.levelTimeRemaining = gameState.tournament.levelDuration * 60;
    
    // 更新时间显示
    updateLevelTimeDisplay();
    
    // 设置计时器
    gameState.tournament.levelTimer = setInterval(() => {
        // 减少剩余时间
        gameState.tournament.levelTimeRemaining--;
        
        // 更新显示
        updateLevelTimeDisplay();
        
        // 如果时间到，升级盲注
        if (gameState.tournament.levelTimeRemaining <= 0) {
            increaseBlinds();
        }
    }, 1000);
}

// 更新级别时间显示
function updateLevelTimeDisplay() {
    const minutes = Math.floor(gameState.tournament.levelTimeRemaining / 60);
    const seconds = gameState.tournament.levelTimeRemaining % 60;
    
    // 格式化显示时间
    const timeDisplay = `${minutes}:${seconds < 10 ? '0' + seconds : seconds}`;
    document.getElementById('levelTimeRemaining').textContent = timeDisplay;
    
    // 更新进度条
    const progressPercent = (gameState.tournament.levelTimeRemaining / (gameState.tournament.levelDuration * 60)) * 100;
    document.getElementById('timerProgressBar').style.width = `${progressPercent}%`;
    
    // 剩余时间少于1分钟时添加警告样式
    if (gameState.tournament.levelTimeRemaining < 60) {
        document.getElementById('levelTimeRemaining').classList.add('warning');
    } else {
        document.getElementById('levelTimeRemaining').classList.remove('warning');
    }
}

// 增加盲注
function increaseBlinds() {
    // 增加级别
    gameState.tournament.currentLevel++;
    
    // 基于级别设置新的盲注
    const blindLevels = [
        { small: 5, big: 10 },
        { small: 10, big: 20 },
        { small: 15, big: 30 },
        { small: 20, big: 40 },
        { small: 25, big: 50 },
        { small: 50, big: 100 },
        { small: 75, big: 150 },
        { small: 100, big: 200 },
        { small: 150, big: 300 },
        { small: 200, big: 400 },
        { small: 300, big: 600 },
        { small: 400, big: 800 },
        { small: 500, big: 1000 },
        { small: 750, big: 1500 },
        { small: 1000, big: 2000 }
    ];
    
    // 确保不超出级别范围
    const levelIndex = Math.min(gameState.tournament.currentLevel, blindLevels.length - 1);
    
    // 设置新盲注
    gameState.smallBlind = blindLevels[levelIndex].small;
    gameState.bigBlind = blindLevels[levelIndex].big;
    
    // 更新显示
    document.getElementById('currentLevel').textContent = gameState.tournament.currentLevel + 1;
    document.getElementById('currentBlinds').textContent = `${gameState.smallBlind}/${gameState.bigBlind}`;
    
    // 重置级别时间
    startLevelTimer();
    
    // 播放级别提升音效
    playSound('levelUp');
    
    // 添加级别提升记录
    addHistoryItem('系统', `盲注提升至 ${gameState.smallBlind}/${gameState.bigBlind}`, null, true);
    
    // 显示级别提升通知
    showBlindsUpNotification();
}

// 显示盲注提升通知
function showBlindsUpNotification() {
    // 创建通知元素
    const notification = document.createElement('div');
    notification.className = 'blinds-up-notification';
    
    // 设置通知内容
    notification.innerHTML = `
        <div class="blinds-up-icon">⏱️</div>
        <div class="blinds-up-title">盲注提升!</div>
        <div class="blinds-up-level">级别 ${gameState.tournament.currentLevel + 1}</div>
        <div class="blinds-up-blinds">${gameState.smallBlind}/${gameState.bigBlind}</div>
    `;
    
    // 添加到页面
    document.body.appendChild(notification);
    
    // 动画结束后移除
    notification.addEventListener('animationend', () => {
        notification.remove();
    });
    
    // 高亮盲注显示
    const blindsDisplay = document.getElementById('currentBlinds');
    blindsDisplay.classList.add('new-level');
    
    // 添加锦标赛信息区域高亮
    const tournamentInfo = document.getElementById('tournamentInfo');
    tournamentInfo.classList.add('level-up-highlight');
    
    // 移除高亮效果
    setTimeout(() => {
        blindsDisplay.classList.remove('new-level');
        tournamentInfo.classList.remove('level-up-highlight');
    }, 5000);
}

// 处理玩家被淘汰
function handlePlayerElimination(playerIndex) {
    // 获取被淘汰的玩家
    const player = gameState.players[playerIndex];
    
    // 标记为已淘汰
    player.eliminated = true;
    
    // 添加被淘汰记录
    addHistoryItem('系统', `${player.name} 被淘汰`, null, true);
    
    // 播放淘汰音效
    playSound('elimination');
    
    // 获取玩家位置元素
    const playerElement = document.getElementById(`player${playerIndex}`);
    if (playerElement) {
        playerElement.classList.add('eliminated');
        
        // 创建淘汰爆炸效果
        const explosion = document.createElement('div');
        explosion.className = 'elimination-explosion';
        explosion.style.left = playerElement.style.left;
        explosion.style.top = playerElement.style.top;
        document.body.appendChild(explosion);
        
        // 创建淘汰文字
        const eliminationText = document.createElement('div');
        eliminationText.className = 'elimination-text';
        eliminationText.textContent = '淘汰!';
        eliminationText.style.left = playerElement.style.left;
        eliminationText.style.top = playerElement.style.top;
        document.body.appendChild(eliminationText);
        
        // 动画结束后移除
        setTimeout(() => {
            explosion.remove();
            eliminationText.remove();
        }, 2000);
    }
    
    // 更新锦标赛信息
    const remainingPlayers = gameState.players.filter(p => !p.eliminated).length;
    document.getElementById('playersRemaining').textContent = remainingPlayers;
    
    // 检查锦标赛是否结束
    checkTournamentEnd();
}

// 处理弃牌
function handleFold() {
    console.log("玩家选择弃牌");
    
    // 获取主玩家
    const mainPlayer = gameState.players[0];
    
    // 标记为弃牌
    mainPlayer.folded = true;
    
    // 添加历史记录
    addHistoryItem(mainPlayer.name, 'fold');
    
    // 显示弃牌对话
    const dialogText = getRandomDialog('fold', mainPlayer.name);
    createDialogBubble(dialogText, PLAYER_POSITIONS[0], 0);
    
    // 播放弃牌音效
    playSound('fold');
    
    // 禁用所有按钮
    foldBtn.disabled = true;
    checkBtn.disabled = true;
    callBtn.disabled = true;
    raiseBtn.disabled = true;
    allInBtn.disabled = true;
    
    // 进入下一个玩家
    setTimeout(() => {
        afterPlayerAction();
    }, 1000);
}

// 处理让牌
function handleCheck() {
    console.log("玩家选择让牌");
    
    // 获取主玩家
    const mainPlayer = gameState.players[0];
    
    // 如果不可以让牌，不执行操作
    if (gameState.currentBet > mainPlayer.bet) {
        console.error("不能让牌，必须跟注或弃牌");
        return;
    }
    
    // 添加历史记录
    addHistoryItem(mainPlayer.name, 'check');
    
    // 显示让牌对话
    const dialogText = getRandomDialog('check', mainPlayer.name);
    createDialogBubble(dialogText, PLAYER_POSITIONS[0], 0);
    
    // 播放让牌音效
    playSound('check');
    
    // 禁用所有按钮
    foldBtn.disabled = true;
    checkBtn.disabled = true;
    callBtn.disabled = true;
    raiseBtn.disabled = true;
    allInBtn.disabled = true;
    
    // 进入下一个玩家
        setTimeout(() => {
        afterPlayerAction();
    }, 1000);
}

// 处理跟注
function handleCall() {
    console.log("玩家选择跟注");
    
    // 获取主玩家
    const mainPlayer = gameState.players[0];
    
    // 计算需要跟注的金额
    const callAmount = gameState.currentBet - mainPlayer.bet;
    
    // 如果跟注金额大于玩家筹码，自动全押
    if (callAmount >= mainPlayer.chips) {
        handleAllIn();
        return;
    }
    
    // 更新玩家筹码和下注金额
    mainPlayer.chips -= callAmount;
    mainPlayer.bet += callAmount;
    
    // 更新底池
    gameState.pot += callAmount;
    
    // 添加历史记录
    addHistoryItem(mainPlayer.name, 'call', callAmount);
    
    // 显示跟注对话
    const dialogText = getRandomDialog('call', mainPlayer.name);
    createDialogBubble(dialogText, PLAYER_POSITIONS[0], 0);
    
    // 创建筹码动画
    createChipAnimation(
        { left: '50%', top: '85%' },
        { left: '50%', top: '50%' },
        callAmount
    );
    
    // 播放跟注音效
    playSound('call');
    
    // 更新显示
    updateGameDisplay();
    
    // 禁用所有按钮
    foldBtn.disabled = true;
    checkBtn.disabled = true;
    callBtn.disabled = true;
    raiseBtn.disabled = true;
    allInBtn.disabled = true;
    
    // 进入下一个玩家
    setTimeout(() => {
        afterPlayerAction();
    }, 1000);
}

// 处理加注
function handleRaise() {
    console.log("玩家选择加注");
    
    // 获取主玩家
    const mainPlayer = gameState.players[0];
    
    // 获取加注金额
    const raiseAmount = parseInt(betSlider.value);
    
    // 验证加注金额
    if (isNaN(raiseAmount) || raiseAmount <= gameState.currentBet || raiseAmount > mainPlayer.chips + mainPlayer.bet) {
        console.error("无效的加注金额");
        return;
    }
    
    // 计算实际需要支付的筹码
    const actualPay = raiseAmount - mainPlayer.bet;
    
    // 更新玩家筹码和下注金额
    mainPlayer.chips -= actualPay;
    mainPlayer.bet = raiseAmount;
    
    // 更新当前下注
    gameState.currentBet = raiseAmount;
    
    // 更新底池
    gameState.pot += actualPay;
    
    // 添加历史记录
    addHistoryItem(mainPlayer.name, 'raise', actualPay);
    
    // 显示加注对话
    const dialogText = getRandomDialog('raise', mainPlayer.name);
    createDialogBubble(dialogText, PLAYER_POSITIONS[0], 0);
    
    // 创建筹码动画
    createChipAnimation(
        { left: '50%', top: '85%' },
        { left: '50%', top: '50%' },
        actualPay
    );
    
    // 播放加注音效
    playSound('raise');
    
    // 更新显示
    updateGameDisplay();
    
    // 禁用所有按钮
    foldBtn.disabled = true;
    checkBtn.disabled = true;
    callBtn.disabled = true;
    raiseBtn.disabled = true;
    allInBtn.disabled = true;
    
    // 进入下一个玩家
    setTimeout(() => {
        afterPlayerAction();
    }, 1000);
}

// 处理全押
function handleAllIn() {
    console.log("玩家选择全押");
    
    // 获取主玩家
    const mainPlayer = gameState.players[0];
    
    // 如果已经没有筹码，不执行操作
    if (mainPlayer.chips <= 0) {
        console.error("没有筹码可以全押");
        return;
    }
    
    // 计算全押金额
    const allInAmount = mainPlayer.chips;
    
    // 更新玩家筹码和下注金额
    mainPlayer.chips = 0;
    mainPlayer.bet += allInAmount;
    mainPlayer.isAllIn = true;
    
    // 如果全押金额超过当前下注，更新当前下注
    if (mainPlayer.bet > gameState.currentBet) {
        gameState.currentBet = mainPlayer.bet;
    }
    
    // 更新底池
    gameState.pot += allInAmount;
    
    // 添加历史记录
    addHistoryItem(mainPlayer.name, 'allIn', allInAmount);
    
    // 显示全押对话
    const dialogText = getRandomDialog('allIn', mainPlayer.name);
    createDialogBubble(dialogText, PLAYER_POSITIONS[0], 0);
    
    // 创建筹码动画
    createChipAnimation(
        { left: '50%', top: '85%' },
        { left: '50%', top: '50%' },
        allInAmount
    );
    
    // 播放全押音效
    playSound('raise');
    
    // 更新显示
    updateGameDisplay();
    
    // 禁用所有按钮
    foldBtn.disabled = true;
    checkBtn.disabled = true;
    callBtn.disabled = true;
    raiseBtn.disabled = true;
    allInBtn.disabled = true;
    
    // 进入下一个玩家
    setTimeout(() => {
        afterPlayerAction();
    }, 1000);
}

// 增强AI决策中的位置因素
function makeAIDecision(playerIndex) {
    // 计算手牌强度
    const player = gameState.players[playerIndex];
    let handStrength = calculateHandStrength(player.hand, gameState.communityCards);
    
    // 增加位置因素
    const positions = getRelativePositions(gameState.dealerPosition, gameState.players.length);
    const positionFactor = evaluatePosition(positions[playerIndex]);
    handStrength *= (1 + positionFactor);
    
    // 继续使用第一个makeAIDecision函数的逻辑
    return gameState.players[playerIndex].folded ? 'fold' : 'check';
}

function evaluatePosition(position) {
    // 晚位置加成，早位置减弱
    if (position === 'late') return 0.15;
    if (position === 'middle') return 0.05;
    if (position === 'early') return -0.05;
    if (position === 'blinds') return -0.1;
    return 0;
}

// 根据庄家位置计算相对位置
function getRelativePositions(dealerPosition, playerCount) {
    const positions = [];
    for (let i = 0; i < playerCount; i++) {
        // 计算相对位置
        const relativePosition = (i - dealerPosition - 3 + playerCount) % playerCount;
        const positionPercentile = relativePosition / playerCount;
        
        // 根据位置百分比分配位置名称
        let positionName;
        if (i === (dealerPosition + 1) % playerCount || i === (dealerPosition + 2) % playerCount) {
            positionName = 'blinds'; // 小盲和大盲
        } else if (positionPercentile < 0.3) {
            positionName = 'early'; // 早位
        } else if (positionPercentile < 0.7) {
            positionName = 'middle'; // 中位
        } else {
            positionName = 'late'; // 晚位
        }
        positions[i] = positionName;
    }
    return positions;
}

// 1. 添加边池计算逻辑
function calculateSidePots() {
    // 获取所有all-in玩家及其下注金额
    const allInPlayers = gameState.players.filter(p => p.isAllIn && !p.folded);
    if (allInPlayers.length === 0) return [{ amount: gameState.pot, eligiblePlayers: gameState.players.filter(p => !p.folded) }];
    
    // 按下注量排序
    allInPlayers.sort((a, b) => a.bet - b.bet);
    
    const pots = [];
    let previousBet = 0;
    let remainingPot = gameState.pot;
    
    // 为每个all-in玩家计算边池
    for (const allInPlayer of allInPlayers) {
        const currentBet = allInPlayer.bet;
        const betDifference = currentBet - previousBet;
        
        if (betDifference > 0) {
            // 计算此层边池的大小
            const activePlayersCount = gameState.players.filter(p => !p.folded && p.bet >= currentBet).length;
            const potAmount = betDifference * activePlayersCount;
            
            // 确定有资格赢得此边池的玩家
            const eligiblePlayers = gameState.players.filter(p => !p.folded && p.bet >= currentBet);
            
            pots.push({ amount: potAmount, eligiblePlayers });
            remainingPot -= potAmount;
            previousBet = currentBet;
        }
    }
    
    // 添加主池
    if (remainingPot > 0) {
        const eligiblePlayers = gameState.players.filter(p => !p.folded);
        pots.push({ amount: remainingPot, eligiblePlayers });
    }
    
    return pots;
}

// 2. 改进手牌评估，确保钢板顺被正确识别
function evaluateHandRank(cards) {
    // 现有代码...
    
    // 检查同花顺
    if (hasStraight && hasFlush) {
        // 检查是否为钢板顺(A-2-3-4-5)
        const hasAce = values.includes(14); // Ace
        const hasFive = values.includes(5);
        const hasSteelWheel = hasAce && hasFive && 
            values.includes(2) && values.includes(3) && values.includes(4);
            
        if (hasSteelWheel) {
            return { type: 'straight_flush', value: 5 }; // 5高的同花顺
        }
        
        // 常规同花顺逻辑...
    }
    
    // 现有代码继续...
}

// 3. 增加统计和分析功能
function calculateWinningOdds() {
    if (gameState.gamePhase === 'preflop') return preCalcOdds();
    
    const playerHand = gameState.players[0].hand;
    const communityCards = gameState.communityCards;
    const remainingCards = 5 - communityCards.length;
    
    // 使用蒙特卡洛模拟计算概率
    let wins = 0;
    const simulations = 1000;
    
    for (let i = 0; i < simulations; i++) {
        // 创建模拟牌组
        const deck = createDeck().filter(card => 
            !communityCards.some(c => c.suit === card.suit && c.value === card.value) &&
            !playerHand.some(c => c.suit === card.suit && c.value === card.value)
        );
        
        // 洗牌
        const shuffledDeck = shuffle([...deck]);
        
        // 取出剩余的公共牌
        const simulatedCommunity = [...communityCards];
        for (let j = 0; j < remainingCards; j++) {
            simulatedCommunity.push(shuffledDeck.pop());
        }
        
        // 为每个AI玩家生成手牌
        const playerHands = [];
        for (let j = 1; j < gameState.players.length; j++) {
            if (!gameState.players[j].folded) {
                playerHands.push([shuffledDeck.pop(), shuffledDeck.pop()]);
            } else {
                playerHands.push(null);
            }
        }
        
        // 评估手牌
        const mainPlayerRank = evaluateHandRank([...playerHand, ...simulatedCommunity]);
        let isWinner = true;
        
        for (let j = 0; j < playerHands.length; j++) {
            if (playerHands[j]) {
                const opponentRank = evaluateHandRank([...playerHands[j], ...simulatedCommunity]);
                if (compareHandRanks(opponentRank, mainPlayerRank) > 0) {
                    isWinner = false;
                    break;
                }
            }
        }
        
        if (isWinner) wins++;
    }
    
    return (wins / simulations) * 100;
}
