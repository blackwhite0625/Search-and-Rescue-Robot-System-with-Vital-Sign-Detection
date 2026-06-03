/**
 * 搜救機器人 — 探索地圖繪製模組
 * ================================
 * 即時繪製車子行進路線、障礙物、人員標記。
 * 自動追蹤車子位置（視窗跟隨），支援 HiDPI 銳利渲染。
 *
 * 用法：
 *   const map = new ExplorationMap('map-canvas');
 *   // 在 status polling 中：
 *   map.update(data.heat_map);
 */

class ExplorationMap {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        if (!this.canvas) return;
        this.ctx = this.canvas.getContext('2d');
        this._prevDataHash = '';
        this._setupHiDPI();
    }

    /** HiDPI 銳利渲染（Retina 螢幕不模糊） */
    _setupHiDPI() {
        const dpr = window.devicePixelRatio || 1;
        const rect = this.canvas.getBoundingClientRect();
        this.canvas.width = rect.width * dpr;
        this.canvas.height = rect.height * dpr;
        this.ctx.scale(dpr, dpr);
        this._cssW = rect.width;
        this._cssH = rect.height;
    }

    /** 主更新入口（每次 polling 呼叫） */
    update(hm) {
        if (!this.ctx || !hm) return;

        // 簡易髒檢查：資料沒變就不重繪
        const hash = `${hm.robot_grid_x},${hm.robot_grid_y},${hm.heading_deg},${(hm.path||[]).length},${(hm.obstacles||[]).length},${(hm.persons||[]).length},${(hm.scanned||[]).length}`;
        if (hash === this._prevDataHash) return;
        this._prevDataHash = hash;

        const ctx = this.ctx;
        const gs = hm.grid_size || 40;
        const w = this._cssW;
        const h = this._cssH;
        const cell = w / gs;

        // 清除
        ctx.fillStyle = '#1e1e1e';
        ctx.fillRect(0, 0, w, h);

        this._drawGrid(ctx, gs, cell, w, h);
        this._drawScannedArea(ctx, hm, gs, cell);
        this._drawPath(ctx, hm.path || [], cell);
        this._drawObstacles(ctx, hm.obstacles || [], cell);
        this._drawPersons(ctx, hm.persons || [], cell);
        this._drawRobot(ctx, hm, cell);
        this._drawInfo(ctx, hm, w);
    }

    /** 背景格線 */
    _drawGrid(ctx, gs, cell, w, h) {
        ctx.strokeStyle = 'rgba(255,255,255,0.03)';
        ctx.lineWidth = 0.5;
        for (let i = 0; i <= gs; i++) {
            const pos = i * cell;
            ctx.beginPath(); ctx.moveTo(pos, 0); ctx.lineTo(pos, h); ctx.stroke();
            ctx.beginPath(); ctx.moveTo(0, pos); ctx.lineTo(w, pos); ctx.stroke();
        }
        // 中心十字線（起點標記）
        const cx = (gs / 2) * cell, cy = (gs / 2) * cell;
        ctx.strokeStyle = 'rgba(255,255,255,0.08)';
        ctx.lineWidth = 1;
        ctx.setLineDash([3, 3]);
        ctx.beginPath(); ctx.moveTo(cx, 0); ctx.lineTo(cx, h); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(0, cy); ctx.lineTo(w, cy); ctx.stroke();
        ctx.setLineDash([]);
    }

    /** 已掃描區域（淡色底） — 用 path 點周圍 2 格模擬 */
    _drawScannedArea(ctx, hm, gs, cell) {
        // 優先用後端回傳的精確掃描資料
        const scanned = hm.scanned || [];
        if (scanned.length > 0) {
            ctx.fillStyle = 'rgba(68, 170, 170, 0.12)';
            for (const [gx, gy] of scanned) {
                ctx.fillRect(gx * cell, gy * cell, cell, cell);
            }
            return;
        }
        // 備援：用路徑估算（舊邏輯）
        const path = hm.path || [];
        if (path.length === 0) return;
        ctx.fillStyle = 'rgba(68, 170, 170, 0.08)';
        const seen = new Set();
        const r = 2;
        for (const [gx, gy] of path) {
            for (let dx = -r; dx <= r; dx++) {
                for (let dy = -r; dy <= r; dy++) {
                    if (dx * dx + dy * dy > r * r) continue;
                    const nx = gx + dx, ny = gy + dy;
                    const key = nx * 1000 + ny;
                    if (seen.has(key) || nx < 0 || ny < 0 || nx >= gs || ny >= gs) continue;
                    seen.add(key);
                    ctx.fillRect(nx * cell, ny * cell, cell, cell);
                }
            }
        }
    }

    /** 行進路線（漸層青色，越新越亮） */
    _drawPath(ctx, path, cell) {
        if (path.length < 2) return;

        const total = path.length;
        ctx.lineWidth = 2.5;
        ctx.lineJoin = 'round';
        ctx.lineCap = 'round';

        // 分段漸層：舊路線暗，新路線亮
        for (let i = 1; i < total; i++) {
            const alpha = 0.15 + 0.85 * (i / total);
            ctx.strokeStyle = `rgba(68, 190, 190, ${alpha})`;
            ctx.beginPath();
            ctx.moveTo(path[i - 1][0] * cell + cell / 2, path[i - 1][1] * cell + cell / 2);
            ctx.lineTo(path[i][0] * cell + cell / 2, path[i][1] * cell + cell / 2);
            ctx.stroke();
        }

        // 起點標記（小圓圈）
        const sx = path[0][0] * cell + cell / 2, sy = path[0][1] * cell + cell / 2;
        ctx.fillStyle = 'rgba(68, 170, 170, 0.5)';
        ctx.beginPath(); ctx.arc(sx, sy, 4, 0, Math.PI * 2); ctx.fill();
        ctx.fillStyle = '#1e1e1e';
        ctx.beginPath(); ctx.arc(sx, sy, 2, 0, Math.PI * 2); ctx.fill();
    }

    /** 障礙物（橘色三角錐 + 白邊） */
    _drawObstacles(ctx, obstacles, cell) {
        obstacles.forEach(o => {
            const ox = o[0] * cell + cell / 2;
            const oy = o[1] * cell + cell / 2;
            const s = cell * 0.5;

            // 外框
            ctx.strokeStyle = 'rgba(230, 126, 34, 0.6)';
            ctx.lineWidth = 1;
            ctx.fillStyle = '#e67e22';
            ctx.beginPath();
            ctx.moveTo(ox, oy - s);
            ctx.lineTo(ox - s * 0.8, oy + s * 0.5);
            ctx.lineTo(ox + s * 0.8, oy + s * 0.5);
            ctx.closePath();
            ctx.fill();
            ctx.stroke();

            // 驚嘆號
            ctx.fillStyle = '#fff';
            ctx.font = `bold ${Math.max(8, cell * 0.35)}px sans-serif`;
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText('!', ox, oy + s * 0.05);
        });
    }

    /** 人員標記（人形圖示） */
    _drawPersons(ctx, persons, cell) {
        persons.forEach(p => {
            const px = p.gx * cell + cell / 2;
            const py = p.gy * cell + cell / 2;
            const r = cell * 0.35;
            const color = p.status === 'rescued' ? '#e74c3c' : '#f39c12';

            // 光暈
            ctx.fillStyle = color + '30';
            ctx.beginPath(); ctx.arc(px, py, r * 2, 0, Math.PI * 2); ctx.fill();

            // 頭
            ctx.fillStyle = color;
            ctx.beginPath(); ctx.arc(px, py - r * 1.1, r * 0.5, 0, Math.PI * 2); ctx.fill();

            // 身體
            ctx.strokeStyle = color;
            ctx.lineWidth = 2;
            ctx.beginPath(); ctx.moveTo(px, py - r * 0.6); ctx.lineTo(px, py + r * 0.5); ctx.stroke();

            // 雙手
            ctx.beginPath(); ctx.moveTo(px - r * 0.7, py - r * 0.1); ctx.lineTo(px + r * 0.7, py - r * 0.1); ctx.stroke();

            // 雙腳
            ctx.beginPath(); ctx.moveTo(px, py + r * 0.5); ctx.lineTo(px - r * 0.5, py + r * 1.1); ctx.stroke();
            ctx.beginPath(); ctx.moveTo(px, py + r * 0.5); ctx.lineTo(px + r * 0.5, py + r * 1.1); ctx.stroke();
        });
    }

    /** 車子（綠色箭頭 + 方向扇形） */
    _drawRobot(ctx, hm, cell) {
        const rx = hm.robot_grid_x * cell + cell / 2;
        const ry = hm.robot_grid_y * cell + cell / 2;
        const angle = ((hm.heading_deg || 0)) * Math.PI / 180;
        const len = cell * 0.9;

        ctx.save();
        ctx.translate(rx, ry);
        ctx.rotate(angle);

        // 視野扇形（半透明）
        ctx.fillStyle = 'rgba(46, 204, 113, 0.1)';
        ctx.beginPath();
        ctx.moveTo(0, 0);
        ctx.arc(0, 0, cell * 3, -Math.PI / 3 - Math.PI / 2, Math.PI / 3 - Math.PI / 2);
        ctx.closePath();
        ctx.fill();

        // 箭頭本體
        ctx.fillStyle = '#2ecc71';
        ctx.strokeStyle = '#27ae60';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(0, -len);
        ctx.lineTo(-len * 0.35, len * 0.25);
        ctx.lineTo(0, len * 0.1);
        ctx.lineTo(len * 0.35, len * 0.25);
        ctx.closePath();
        ctx.fill();
        ctx.stroke();

        // 中心白點
        ctx.fillStyle = '#fff';
        ctx.beginPath(); ctx.arc(0, 0, 2.5, 0, Math.PI * 2); ctx.fill();

        ctx.restore();
    }

    /** 右下角資訊 */
    _drawInfo(ctx, hm, w) {
        const cov = hm.coverage || 0;
        const obs = (hm.obstacles || []).length;
        const ppl = (hm.persons || []).length;
        const txt = `${cov}% | ${obs} obs | ${ppl} ppl`;

        ctx.fillStyle = 'rgba(0,0,0,0.5)';
        ctx.fillRect(w - 130, this._cssH - 18, 130, 18);
        ctx.fillStyle = 'rgba(255,255,255,0.6)';
        ctx.font = '10px Inter, sans-serif';
        ctx.textAlign = 'right';
        ctx.textBaseline = 'bottom';
        ctx.fillText(txt, w - 4, this._cssH - 4);
    }
}
