from PIL import Image, ImageDraw, ImageFont

def make_sota_table():
    W, H = 1440, 640
    img = Image.new('RGB', (W, H), 'white')
    d = ImageDraw.Draw(img)

    C_TEAL     = (46, 122, 153)
    C_ROW_ALT  = (232, 244, 249)
    C_NAVY     = (24, 58, 95)
    C_SEP      = (185, 210, 225)
    C_TEXT     = (42, 42, 42)
    C_WHITE    = (255, 255, 255)

    def font(size):
        for p in [
            'C:/Windows/Fonts/msyh.ttc',
            'C:/Windows/Fonts/simhei.ttf',
            'C:/Windows/Fonts/simsun.ttc',
        ]:
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
        return ImageFont.load_default()

    F_TITLE  = font(34)
    F_HEADER = font(19)
    F_BODY   = font(15)
    F_BOLD   = font(16)

    MAR = 55
    TY  = 42

    # Title
    d.rectangle([MAR, TY, MAR + 7, TY + 42], fill=C_TEAL)
    d.text((MAR + 22, TY + 2), 'S O T A  对 比', font=F_TITLE, fill=C_NAVY)

    # Separator
    d.line([(MAR, TY + 64), (W - MAR, TY + 64)], fill=C_SEP, width=1)

    TX   = MAR
    TBY  = TY + 86
    ROW_H = 90

    # Column widths: 类别 | 代表方法 | 来源 | 对比方式
    COL_W = [185, 170, 162, W - MAR * 2 - 185 - 170 - 162]

    headers = ['类别', '代表方法', '来源', '对比方式']
    rows = [
        ['进化多目标 NAS', 'NSGA-NetV2', "Lu et al.\nECCV '20",
         '同搜索空间/数据/预算重跑，比 HV、C-metric、Pareto 图\n（与本方案同范式，最 apples-to-apples 的方法级对比）'],
        ['硬件感知可微 NAS', 'ProxylessNAS', "Cai et al.\nICLR '19",
         '公开代码，以 RK3566 实测延迟为目标在本数据上重搜'],
        ['高效超网 NAS', 'Once-for-All', "Cai et al.\nICLR '20",
         '用其超网在 RK3566 延迟约束下检索子网，retrain 后比'],
        ['边缘极小模型 NAS', 'MCUNet / TinyNAS', "Lin et al.\nNeurIPS '20",
         '迁移其搜索流程/已发布架构到本数据，retrain + 上板测延迟\n（直接对标"极小模型"卖点）'],
    ]

    def draw_cell(x, y, w, h, text, fnt, color=C_TEXT, center=False, bg=None):
        if bg:
            d.rectangle([x, y, x + w, y + h], fill=bg)
        PAD = 16
        lines = text.split('\n')
        line_hs = []
        for ln in lines:
            bb = d.textbbox((0, 0), ln, font=fnt)
            line_hs.append(bb[3] - bb[1])
        GAP = 6
        total_h = sum(line_hs) + GAP * (len(lines) - 1)
        sy = y + max((h - total_h) // 2, PAD // 2)
        for ln, lh in zip(lines, line_hs):
            if center:
                bb = d.textbbox((0, 0), ln, font=fnt)
                lw = bb[2] - bb[0]
                d.text((x + (w - lw) // 2, sy), ln, font=fnt, fill=color)
            else:
                d.text((x + PAD, sy), ln, font=fnt, fill=color)
            sy += lh + GAP

    # Header row
    x = TX
    for h_text, cw in zip(headers, COL_W):
        d.rectangle([x, TBY, x + cw, TBY + ROW_H], fill=C_TEAL)
        draw_cell(x, TBY, cw, ROW_H, h_text, F_HEADER, color=C_WHITE, center=True)
        x += cw

    # Data rows
    for ri, row in enumerate(rows):
        ry = TBY + ROW_H + ri * ROW_H
        bg = C_WHITE if ri % 2 == 0 else C_ROW_ALT
        x = TX
        for ci, (cell, cw) in enumerate(zip(row, COL_W)):
            center = ci in (0, 2)
            fnt = F_BOLD if ci == 1 else F_BODY
            draw_cell(x, ry, cw, ROW_H, cell, fnt, color=C_TEXT, center=center, bg=bg)
            x += cw

    # Horizontal grid lines
    total_w = sum(COL_W)
    for ri in range(len(rows) + 2):
        y = TBY + ri * ROW_H
        d.line([(TX, y), (TX + total_w, y)], fill=C_SEP, width=1)

    # Outer border
    d.rectangle([TX, TBY, TX + total_w, TBY + (len(rows) + 1) * ROW_H],
                outline=C_SEP, width=1)

    out = 'd:/NAS项目/sota_comparison.png'
    img.save(out)
    print(f'Saved: {out}')

make_sota_table()
