import { BoxGeometry, Color, Group, Matrix4, Mesh, MeshStandardMaterial, Quaternion, Vector3, type BufferGeometry } from 'three';

/**
 * PedBuilder: a six-part low-poly human (head, torso, two arms, two legs) built
 * from boxes, plus a procedural walk/run/aim pose. The Player uses the Group-based
 * rig (`buildPed` + `animatePed`); the ped system renders the same rig through
 * InstancedMeshes using `computePedPartMatrices`, which mirrors `animatePed`'s math.
 *
 * Rig convention: root origin at the feet; +Z is the ped's forward direction.
 */
export interface PedPalette {
  skin: number;
  hair: number;
  shirt: number;
  pants: number;
  shoes: number;
}

export interface PedPose {
  /** Walk cycle phase in radians; advance by speed * dt * stride rate. */
  phase: number;
  /** 0 = standing, 1 = full sprint; scales limb swing and bob. */
  speed01: number;
  /** Arms forward for aiming. */
  aiming: boolean;
  /** Both arms straight up (cower). */
  handsUp: boolean;
  /** 0 = upright, 1 = lying on the back (death tip-over). */
  fallen: number;
}

export interface PedRig {
  root: Group;
  head: Mesh;
  torso: Mesh;
  armL: Group;
  armR: Group;
  legL: Group;
  legR: Group;
  /** Meshes in draw order: head, torso, armL, armR, legL, legR. */
  meshes: Mesh[];
  palette: PedPalette;
}

export const PED_DIMS = {
  height: 1.8,
  head: { w: 0.26, h: 0.28, d: 0.26 },
  torso: { w: 0.44, h: 0.62, d: 0.24 },
  arm: { w: 0.12, h: 0.62, d: 0.12 },
  leg: { w: 0.16, h: 0.86, d: 0.16 },
  hipY: 0.86,
  shoulderY: 1.46,
  headY: 1.64,
} as const;

export const PED_PALETTES: Record<string, PedPalette> = {
  lucia: { skin: 0xb9805a, hair: 0x1d1410, shirt: 0xf2f2f2, pants: 0x2c3444, shoes: 0x1a1a1a },
  jason: { skin: 0xd9a77c, hair: 0x5a3b22, shirt: 0x274c8f, pants: 0x8b7355, shoes: 0x3a2a1a },
  beachgoer: { skin: 0xe0b08a, hair: 0xd8b56b, shirt: 0xff6fa3, pants: 0x35e6ff, shoes: 0xfff0d0 },
  business: { skin: 0xc48b62, hair: 0x2a1c14, shirt: 0x1f1f2a, pants: 0x1a1a24, shoes: 0x101010 },
  worker: { skin: 0x8f5a3a, hair: 0x1a1210, shirt: 0xff8a1f, pants: 0x3a4a7a, shoes: 0x3a2a1a },
  cop: { skin: 0xd1a37d, hair: 0x2a1c14, shirt: 0x2b3b8f, pants: 0x1a2350, shoes: 0x0e0e0e },
};

let sharedGeometries: { head: BufferGeometry; torso: BufferGeometry; arm: BufferGeometry; leg: BufferGeometry } | null = null;

/** Shared part geometries (origin at the joint for limbs, at center for head/torso). */
export function pedPartGeometries(): { head: BufferGeometry; torso: BufferGeometry; arm: BufferGeometry; leg: BufferGeometry } {
  if (sharedGeometries) return sharedGeometries;
  const d = PED_DIMS;
  const arm = new BoxGeometry(d.arm.w, d.arm.h, d.arm.d);
  arm.translate(0, -d.arm.h / 2, 0);
  const leg = new BoxGeometry(d.leg.w, d.leg.h, d.leg.d);
  leg.translate(0, -d.leg.h / 2, 0);
  sharedGeometries = {
    head: new BoxGeometry(d.head.w, d.head.h, d.head.d),
    torso: new BoxGeometry(d.torso.w, d.torso.h, d.torso.d),
    arm,
    leg,
  };
  return sharedGeometries;
}

function material(color: number): MeshStandardMaterial {
  return new MeshStandardMaterial({ color: new Color(color), roughness: 0.85, metalness: 0.0 });
}

export function buildPed(palette: PedPalette): PedRig {
  const g = pedPartGeometries();
  const d = PED_DIMS;
  const root = new Group();

  const torso = new Mesh(g.torso, material(palette.shirt));
  torso.position.y = d.hipY + d.torso.h / 2;

  const head = new Mesh(g.head, material(palette.skin));
  head.position.y = d.headY;
  // Hair cap: a slightly wider, thin box on top of the head.
  const hair = new Mesh(new BoxGeometry(d.head.w + 0.02, 0.08, d.head.d + 0.02), material(palette.hair));
  hair.position.y = d.head.h / 2 - 0.02;
  head.add(hair);

  const makeArm = (side: number): Group => {
    const pivot = new Group();
    pivot.position.set(side * (d.torso.w / 2 + d.arm.w / 2 + 0.01), d.shoulderY, 0);
    // Sleeve covers the upper 45% of the arm, bare skin the lower 55%.
    const sleeve = new Mesh(new BoxGeometry(d.arm.w + 0.02, d.arm.h * 0.45, d.arm.d + 0.02), material(palette.shirt));
    sleeve.position.y = -d.arm.h * 0.225;
    const skin = new Mesh(new BoxGeometry(d.arm.w, d.arm.h * 0.55, d.arm.d), material(palette.skin));
    skin.position.y = -d.arm.h * 0.725;
    pivot.add(sleeve, skin);
    return pivot;
  };
  const makeLeg = (side: number): Group => {
    const pivot = new Group();
    pivot.position.set(side * (d.leg.w / 2 + 0.03), d.hipY, 0);
    const leg = new Mesh(g.leg, material(palette.pants));
    const shoe = new Mesh(new BoxGeometry(d.leg.w + 0.02, 0.1, d.leg.d + 0.1), material(palette.shoes));
    shoe.position.set(0, -d.leg.h + 0.05, 0.03);
    pivot.add(leg, shoe);
    return pivot;
  };

  const armL = makeArm(-1);
  const armR = makeArm(1);
  const legL = makeLeg(-1);
  const legR = makeLeg(1);
  root.add(torso, head, armL, armR, legL, legR);
  const meshes = [head, torso, armL.children[0] as Mesh, armR.children[0] as Mesh, legL.children[0] as Mesh, legR.children[0] as Mesh];
  for (const m of root.children) {
    m.castShadow = true;
    m.receiveShadow = false;
    m.traverse((c) => {
      c.castShadow = true;
    });
  }
  return { root, head, torso, armL, armR, legL, legR, meshes, palette };
}

/** Applies a pose to the rig. Cheap enough to call every frame for the player. */
export function animatePed(rig: PedRig, pose: PedPose): void {
  const swing = 0.15 + 0.75 * pose.speed01;
  const s = Math.sin(pose.phase) * swing;
  const legLift = Math.max(0, Math.cos(pose.phase)) * 0.15 * pose.speed01;
  rig.legL.rotation.x = s;
  rig.legR.rotation.x = -s;
  rig.legL.rotation.z = 0;
  rig.legR.rotation.z = 0;

  if (pose.handsUp) {
    rig.armL.rotation.set(Math.PI - 0.15, 0, 0.25);
    rig.armR.rotation.set(Math.PI - 0.15, 0, -0.25);
  } else if (pose.aiming) {
    rig.armR.rotation.set(Math.PI / 2 - 0.1, 0, -0.12);
    rig.armL.rotation.set(Math.PI / 2 - 0.25, 0, 0.35);
  } else {
    rig.armL.rotation.set(-s * 0.9, 0, 0.08);
    rig.armR.rotation.set(s * 0.9, 0, -0.08);
  }

  const bob = Math.abs(Math.sin(pose.phase)) * 0.04 * pose.speed01 + legLift * 0.3;
  const lean = 0.08 * pose.speed01;
  rig.torso.position.y = PED_DIMS.hipY + PED_DIMS.torso.h / 2 + bob;
  rig.head.position.y = PED_DIMS.headY + bob;
  rig.armL.position.y = PED_DIMS.shoulderY + bob;
  rig.armR.position.y = PED_DIMS.shoulderY + bob;
  rig.torso.rotation.x = lean;
  rig.head.rotation.x = -lean * 0.5;

  // Death tip-over: rotate the whole rig backwards around the feet.
  rig.root.rotation.x = -pose.fallen * (Math.PI / 2);
  rig.root.position.y = pose.fallen * 0.15;
}

export function setPedPalette(rig: PedRig, palette: PedPalette): void {
  rig.palette = palette;
  (rig.head.material as MeshStandardMaterial).color.set(palette.skin);
  (rig.torso.material as MeshStandardMaterial).color.set(palette.shirt);
  for (const arm of [rig.armL, rig.armR]) {
    ((arm.children[0] as Mesh).material as MeshStandardMaterial).color.set(palette.shirt);
    ((arm.children[1] as Mesh).material as MeshStandardMaterial).color.set(palette.skin);
  }
  for (const leg of [rig.legL, rig.legR]) {
    ((leg.children[0] as Mesh).material as MeshStandardMaterial).color.set(palette.pants);
    ((leg.children[1] as Mesh).material as MeshStandardMaterial).color.set(palette.shoes);
  }
}

/** Instanced part order produced by `computePedPartMatrices`. */
export const PED_PART_NAMES = ['head', 'hair', 'torso', 'armL', 'armR', 'legL', 'legR'] as const;
export type PedPartName = (typeof PED_PART_NAMES)[number];
export const PED_PART_COUNT = PED_PART_NAMES.length;

/** Hair cap dimensions (matches the box added in `buildPed`). */
export const PED_HAIR = { w: PED_DIMS.head.w + 0.02, h: 0.08, d: PED_DIMS.head.d + 0.02, y: PED_DIMS.head.h / 2 - 0.02 } as const;

const X_AXIS = new Vector3(1, 0, 0);
const Y_AXIS = new Vector3(0, 1, 0);
const ONE = new Vector3(1, 1, 1);
const mRoot = new Matrix4();
const mLocal = new Matrix4();
const qRoot = new Quaternion();
const qTmp = new Quaternion();
const qPart = new Quaternion();
const vPos = new Vector3();
const vLocal = new Vector3();
const eulerTmp = new Vector3();

function localPart(out: Matrix4, x: number, y: number, z: number, rx: number, ry: number, rz: number): void {
  // Same as Object3D with default Euler order XYZ: q = qx * qy * qz.
  qPart.setFromAxisAngle(X_AXIS, rx);
  if (ry !== 0) {
    qTmp.setFromAxisAngle(Y_AXIS, ry);
    qPart.multiply(qTmp);
  }
  if (rz !== 0) {
    qTmp.set(0, 0, Math.sin(rz / 2), Math.cos(rz / 2));
    qPart.multiply(qTmp);
  }
  vLocal.set(x, y, z);
  mLocal.compose(vLocal, qPart, ONE);
  out.multiplyMatrices(mRoot, mLocal);
}

/**
 * Pure-math twin of `animatePed` for instanced rendering. Writes seven world matrices
 * (order: `PED_PART_NAMES`) into `out`. `heading` yaws the rig; `pose.fallen` tips it
 * backwards around its own hips. When `rotation` is given it replaces heading + tip-over
 * entirely (ragdolled peds follow their rigid body), and `position` is still the feet.
 */
export function computePedPartMatrices(pose: PedPose, heading: number, position: Vector3, out: Matrix4[], rotation?: Quaternion): void {
  const d = PED_DIMS;
  if (rotation) {
    qRoot.copy(rotation);
    vPos.copy(position);
  } else {
    qRoot.setFromAxisAngle(Y_AXIS, heading);
    if (pose.fallen > 0) {
      qTmp.setFromAxisAngle(X_AXIS, -pose.fallen * (Math.PI / 2));
      qRoot.multiply(qTmp);
    }
    vPos.set(position.x, position.y + pose.fallen * 0.15, position.z);
  }
  mRoot.compose(vPos, qRoot, ONE);

  const swing = 0.15 + 0.75 * pose.speed01;
  const s = Math.sin(pose.phase) * swing;
  const legLift = Math.max(0, Math.cos(pose.phase)) * 0.15 * pose.speed01;
  const bob = Math.abs(Math.sin(pose.phase)) * 0.04 * pose.speed01 + legLift * 0.3;
  const lean = 0.08 * pose.speed01;

  // head, hair
  localPart(out[0], 0, d.headY + bob, 0, -lean * 0.5, 0, 0);
  vLocal.set(0, PED_HAIR.y, 0);
  mLocal.makeTranslation(vLocal.x, vLocal.y, vLocal.z);
  out[1].multiplyMatrices(out[0], mLocal);
  // torso
  localPart(out[2], 0, d.hipY + d.torso.h / 2 + bob, 0, lean, 0, 0);
  // arms
  const armX = d.torso.w / 2 + d.arm.w / 2 + 0.01;
  if (pose.handsUp) {
    eulerTmp.set(Math.PI - 0.15, 0.25, -0.25);
  } else if (pose.aiming) {
    eulerTmp.set(Math.PI / 2 - 0.25, 0.35, -0.12);
  } else {
    eulerTmp.set(-s * 0.9, 0.08, -0.08);
  }
  const armRx = pose.aiming && !pose.handsUp ? Math.PI / 2 - 0.1 : pose.handsUp ? Math.PI - 0.15 : s * 0.9;
  localPart(out[3], -armX, d.shoulderY + bob, 0, eulerTmp.x, 0, eulerTmp.y);
  localPart(out[4], armX, d.shoulderY + bob, 0, armRx, 0, eulerTmp.z);
  // legs
  const legX = d.leg.w / 2 + 0.03;
  localPart(out[5], -legX, d.hipY, 0, s, 0, 0);
  localPart(out[6], legX, d.hipY, 0, -s, 0, 0);
}

export function disposePed(rig: PedRig): void {
  rig.root.traverse((obj) => {
    const mesh = obj as Mesh;
    if (!mesh.isMesh) return;
    const mat = mesh.material as MeshStandardMaterial;
    mat.dispose();
    // Shared part geometries are kept; only per-rig extras (hair, shoes, sleeves) are freed.
    const g = pedPartGeometries();
    if (mesh.geometry !== g.head && mesh.geometry !== g.torso && mesh.geometry !== g.arm && mesh.geometry !== g.leg) {
      mesh.geometry.dispose();
    }
  });
}
