/**
 * SLAM 地圖渲染器
 * 將 ROS 2 slam_toolbox 的 OccupancyGrid 渲染在 Canvas 上。
 * 資料格式（來自 /status 的 slam_map）：
 *   { width, height, resolution, origin_x, origin_y, walls:[[x,y],...], free:[[x,y],...], seq }
 * pose: [x_m, y_m, yaw_rad]
 * poseHistory: [[x_m, y_m], ...]
 */
class SlamMap {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    if (!this.canvas) return;
    this.ctx = this.canvas.getContext('2d');
    this._lastSeq = -1;
    this._lastPoseSig = null;
    this._lastHistLen = 0;
    this._cachedMap = null;       // 緩存最後一幀 map（處理 dirty-check null）
    this._setupHiDPI();
  }

  _setupHiDPI() {
    const dpr = window.devicePixelRatio || 1;
    const rect = this.canvas.getBoundingClientRect();
    this.canvas.width = rect.width * dpr;
    this.canvas.height = rect.height * dpr;
    this.ctx.scale(dpr, dpr);
    this._w = rect.width;
    this._h = rect.height;
  }

  /**
   * 更新地圖渲染。每當 map/pose/history/lidar 任一改變就重畫。
   */
  update(mapData, pose, poseHistory, lidarPoints, frontiers, reportedVictims) {
    if (!this.canvas || !this.ctx) return;
    // /status 使用 dirty-check get_slam_map()：map 無變化時為 null。
    // 用 cache 保留最後一幀，避免畫面閃爍。
    if (mapData) {
      this._cachedMap = mapData;
    } else if (this._cachedMap) {
      mapData = this._cachedMap;
    } else {
      this._drawNoData();
      return;
    }

    if (this._w === 0 || this._h === 0) {
      this._setupHiDPI();
    }
    if (this._w === 0 || this._h === 0) return;

    // Dirty check：pose 變動或 LiDAR 點有資料就重畫（LiDAR 每次都算新）
    const poseSig = pose && pose.length >= 3
      ? `${pose[0].toFixed(3)},${pose[1].toFixed(3)},${pose[2].toFixed(3)}`
      : 'null';
    const histLen = poseHistory ? poseHistory.length : 0;
    const lidarLen = lidarPoints ? lidarPoints.length : 0;
    const frontierLen = frontiers ? frontiers.length : 0;
    const victimLen = reportedVictims ? reportedVictims.length : 0;
    const mapChanged = mapData.seq !== this._lastSeq;
    const poseChanged = poseSig !== this._lastPoseSig;
    const histChanged = histLen !== this._lastHistLen;
    // 有 LiDAR 點或 frontier 就強制重畫（即時更新）
    if (!mapChanged && !poseChanged && !histChanged && lidarLen === 0 && frontierLen === 0 && victimLen === 0) return;

    this._lastSeq = mapData.seq;
    this._lastPoseSig = poseSig;
    this._lastHistLen = histLen;

    const ctx = this.ctx;
    const w = this._w;
    const h = this._h;

    // 清空
    ctx.fillStyle = '#1a1a2e';
    ctx.fillRect(0, 0, w, h);

    const gw = mapData.width;
    const gh = mapData.height;
    if (gw === 0 || gh === 0) return;

    // 計算縮放
    const scaleX = w / gw;
    const scaleY = h / gh;
    const scale = Math.min(scaleX, scaleY);
    const offsetX = (w - gw * scale) / 2;
    const offsetY = (h - gh * scale) / 2;

    // World → Canvas 座標轉換工具
    const res = mapData.resolution || 0.05;
    const ox = mapData.origin_x || 0;
    const oy = mapData.origin_y || 0;
    const worldToCanvas = (wx, wy) => {
      const gx = (wx - ox) / res;
      const gy = (wy - oy) / res;
      return [
        offsetX + gx * scale,
        offsetY + (gh - 1 - gy) * scale,
      ];
    };

    // 繪製自由空間（淺灰）
    ctx.fillStyle = 'rgba(200, 200, 210, 0.35)';
    const free = mapData.free;
    if (free) {
      for (let i = 0; i < free.length; i++) {
        const gx = free[i][0];
        const gy = free[i][1];
        ctx.fillRect(
          offsetX + gx * scale,
          offsetY + (gh - 1 - gy) * scale,
          Math.max(scale, 1),
          Math.max(scale, 1)
        );
      }
    }

    // 繪製牆壁（紅色）
    ctx.fillStyle = '#e74c3c';
    const walls = mapData.walls;
    if (walls) {
      for (let i = 0; i < walls.length; i++) {
        const gx = walls[i][0];
        const gy = walls[i][1];
        ctx.fillRect(
          offsetX + gx * scale,
          offsetY + (gh - 1 - gy) * scale,
          Math.max(scale, 1.5),
          Math.max(scale, 1.5)
        );
      }
    }

    // 繪製 frontier 點（紫色小方塊 — 未探索邊界）
    if (frontiers && frontiers.length > 0) {
      ctx.fillStyle = 'rgba(200, 120, 255, 0.85)';
      for (let i = 0; i < frontiers.length; i++) {
        const f = frontiers[i];
        const [fx, fy] = worldToCanvas(f[0], f[1]);
        ctx.fillRect(fx - 2, fy - 2, 4, 4);
      }
      // 標記最近 frontier（閃爍金色圓圈 = 當前選擇的目標）
      const nearest = frontiers[0];
      const [tx, ty] = worldToCanvas(nearest[0], nearest[1]);
      const pulse = 0.5 + 0.5 * Math.sin(Date.now() / 180);
      ctx.strokeStyle = `rgba(255, 215, 0, ${0.6 + 0.4 * pulse})`;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(tx, ty, 6 + pulse * 3, 0, Math.PI * 2);
      ctx.stroke();
    }

    // 繪製已通報傷患記憶點（紅色十字 + 安全圈）
    if (reportedVictims && reportedVictims.length > 0) {
      for (let i = 0; i < reportedVictims.length; i++) {
        const v = reportedVictims[i];
        if (typeof v.x !== 'number' || typeof v.y !== 'number') continue;
        const [vx, vy] = worldToCanvas(v.x, v.y);
        const radiusPx = Math.max(8, 0.85 / res * scale);
        ctx.beginPath();
        ctx.arc(vx, vy, radiusPx, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(231, 76, 60, 0.12)';
        ctx.fill();
        ctx.strokeStyle = 'rgba(231, 76, 60, 0.75)';
        ctx.lineWidth = 2;
        ctx.stroke();

        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 4;
        ctx.beginPath();
        ctx.moveTo(vx - 7, vy);
        ctx.lineTo(vx + 7, vy);
        ctx.moveTo(vx, vy - 7);
        ctx.lineTo(vx, vy + 7);
        ctx.stroke();
        ctx.strokeStyle = '#e74c3c';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(vx - 7, vy);
        ctx.lineTo(vx + 7, vy);
        ctx.moveTo(vx, vy - 7);
        ctx.lineTo(vx, vy + 7);
        ctx.stroke();

        ctx.fillStyle = '#fff';
        ctx.font = 'bold 10px monospace';
        ctx.fillText(`#${v.victim_id || '?'}`, vx + 9, vy - 8);
      }
    }

    // 繪製軌跡（藍色 polyline）
    if (poseHistory && poseHistory.length > 1) {
      ctx.strokeStyle = 'rgba(80, 180, 255, 0.85)';
      ctx.lineWidth = 2;
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      // 瞬移偵測閾值（公尺）：超過此距離視為被搬動，斷開線段
      const TELEPORT_THRESHOLD_M = 1.0;
      let currentPath = true;
      ctx.beginPath();
      let prev = null;
      for (let i = 0; i < poseHistory.length; i++) {
        const p = poseHistory[i];
        const [cx, cy] = worldToCanvas(p[0], p[1]);
        if (prev === null) {
          ctx.moveTo(cx, cy);
        } else {
          const dx = p[0] - prev[0];
          const dy = p[1] - prev[1];
          const dist2 = dx * dx + dy * dy;
          if (dist2 > TELEPORT_THRESHOLD_M * TELEPORT_THRESHOLD_M) {
            // 瞬移：結束當前 path，開新的
            ctx.stroke();
            ctx.beginPath();
            ctx.moveTo(cx, cy);
          } else {
            ctx.lineTo(cx, cy);
          }
        }
        prev = p;
      }
      ctx.stroke();

      // 標記軌跡起點（小圓圈）
      const start = poseHistory[0];
      const [sx, sy] = worldToCanvas(start[0], start[1]);
      ctx.fillStyle = 'rgba(80, 180, 255, 0.9)';
      ctx.beginPath();
      ctx.arc(sx, sy, 3, 0, Math.PI * 2);
      ctx.fill();
    }

    // 即時 LiDAR 輪廓線（當前 scan 以 polyline 連接，形成乾淨牆壁線條）
    // 資料格式：[[x_m, y_m], null (段中斷), [x_m, y_m], ...]
    // 相鄰點距離過大 → 分段，避免跨房間亂連線
    if (lidarPoints && lidarPoints.length > 1) {
      ctx.strokeStyle = 'rgba(0, 255, 200, 0.9)';
      ctx.lineWidth = 2;
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      // 斷線閾值（公尺）：相鄰兩個掃描點距離超過此值 → 分段
      const BREAK_DIST_M = 0.30;
      const BREAK_DIST_SQ = BREAK_DIST_M * BREAK_DIST_M;

      ctx.beginPath();
      let prev = null;
      let drawing = false;
      for (let i = 0; i < lidarPoints.length; i++) {
        const p = lidarPoints[i];
        if (p === null) {
          // 後端傳 null 表示該角度無效 → 分段
          if (drawing) ctx.stroke();
          ctx.beginPath();
          drawing = false;
          prev = null;
          continue;
        }
        const [cx, cy] = worldToCanvas(p[0], p[1]);
        if (prev === null) {
          ctx.moveTo(cx, cy);
          drawing = true;
        } else {
          const dx = p[0] - prev[0];
          const dy = p[1] - prev[1];
          if (dx * dx + dy * dy > BREAK_DIST_SQ) {
            // 跳太遠 → 結束當前段，開新段
            if (drawing) ctx.stroke();
            ctx.beginPath();
            ctx.moveTo(cx, cy);
          } else {
            ctx.lineTo(cx, cy);
          }
        }
        prev = p;
      }
      if (drawing) ctx.stroke();

      // 在每個有效點位置畫小點，強化視覺可見度
      ctx.fillStyle = 'rgba(0, 255, 200, 0.7)';
      for (let i = 0; i < lidarPoints.length; i++) {
        const p = lidarPoints[i];
        if (p === null) continue;
        const [cx, cy] = worldToCanvas(p[0], p[1]);
        ctx.fillRect(cx - 1, cy - 1, 2, 2);
      }
    }

    // 繪製機器人位置（高對比大圖示，確保在任何背景都看得見）
    if (pose && pose.length >= 3) {
      const [px, py] = worldToCanvas(pose[0], pose[1]);
      const yaw = pose[2];

      // 外圍脈動光暈（半透明黃色大圓）
      ctx.beginPath();
      ctx.arc(px, py, 22, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(255, 230, 0, 0.2)';
      ctx.fill();

      // 外圓（黑色描邊 + 亮黃填色）
      ctx.beginPath();
      ctx.arc(px, py, 14, 0, Math.PI * 2);
      ctx.fillStyle = '#ffeb00';
      ctx.fill();
      ctx.strokeStyle = '#000';
      ctx.lineWidth = 3;
      ctx.stroke();

      // 中心圓（綠色）
      ctx.beginPath();
      ctx.arc(px, py, 8, 0, Math.PI * 2);
      ctx.fillStyle = '#2ecc71';
      ctx.fill();
      ctx.strokeStyle = '#000';
      ctx.lineWidth = 1.5;
      ctx.stroke();

      // 方向箭頭（白色，黑邊）
      ctx.save();
      ctx.translate(px, py);
      ctx.rotate(-yaw);
      // 視野扇形
      ctx.beginPath();
      ctx.moveTo(0, 0);
      ctx.arc(0, 0, 32, -Math.PI / 4, Math.PI / 4);
      ctx.closePath();
      ctx.fillStyle = 'rgba(255, 230, 0, 0.18)';
      ctx.fill();
      ctx.strokeStyle = 'rgba(255, 230, 0, 0.6)';
      ctx.lineWidth = 1;
      ctx.stroke();
      // 粗箭頭
      ctx.beginPath();
      ctx.moveTo(18, 0);
      ctx.lineTo(4, -6);
      ctx.lineTo(4, 6);
      ctx.closePath();
      ctx.fillStyle = '#ffffff';
      ctx.fill();
      ctx.strokeStyle = '#000';
      ctx.lineWidth = 2;
      ctx.stroke();
      ctx.restore();
    } else {
      // 沒有 pose → 左上角顯示警告
      ctx.fillStyle = 'rgba(231, 76, 60, 0.85)';
      ctx.fillRect(5, 5, 180, 22);
      ctx.fillStyle = '#fff';
      ctx.font = 'bold 12px sans-serif';
      ctx.fillText('⚠ 無機器人位置資料', 12, 21);
    }

    // 資訊覆蓋
    const wallCount = walls ? walls.length : 0;
    const freeCount = free ? free.length : 0;
    const total = gw * gh;
    const coverage = total > 0 ? ((freeCount + wallCount) / total * 100).toFixed(1) : '0.0';
    const histStr = histLen > 0 ? ` | path ${histLen}` : '';
    const frontStr = frontierLen > 0 ? ` | F${frontierLen}` : '';

    ctx.fillStyle = 'rgba(0,0,0,0.5)';
    ctx.fillRect(w - 200, h - 28, 195, 24);
    ctx.fillStyle = '#ccc';
    ctx.font = '11px monospace';
    ctx.textAlign = 'right';
    ctx.fillText(`${coverage}% | ${gw}x${gh} | ${wallCount}w${histStr}${frontStr}`, w - 10, h - 10);
    ctx.textAlign = 'left';

    // Pose 除錯資訊（左上角）— 顯示世界座標與 map origin 讓使用者手動比對
    if (pose && pose.length >= 3) {
      const oxr = (ox || 0).toFixed(2);
      const oyr = (oy || 0).toFixed(2);
      const poseTxt = `Robot: (${pose[0].toFixed(2)}, ${pose[1].toFixed(2)}) m  yaw=${(pose[2] * 180 / Math.PI).toFixed(0)}°`;
      const originTxt = `Map origin: (${oxr}, ${oyr})  res=${(res).toFixed(3)}m  grid=${gw}×${gh}`;
      ctx.fillStyle = 'rgba(0,0,0,0.6)';
      ctx.fillRect(5, 5, 320, 36);
      ctx.fillStyle = '#fff';
      ctx.font = 'bold 11px monospace';
      ctx.fillText(poseTxt, 10, 20);
      ctx.fillStyle = '#aaa';
      ctx.font = '10px monospace';
      ctx.fillText(originTxt, 10, 34);
    }
  }

  _drawNoData() {
    const ctx = this.ctx;
    const w = this._w;
    const h = this._h;
    ctx.fillStyle = '#1a1a2e';
    ctx.fillRect(0, 0, w, h);
    ctx.fillStyle = '#555';
    ctx.font = '14px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('SLAM 未連線', w / 2, h / 2 - 10);
    ctx.font = '11px sans-serif';
    ctx.fillText('等待 ROS 2 SLAM 資料...', w / 2, h / 2 + 10);
    ctx.textAlign = 'left';
  }
}
