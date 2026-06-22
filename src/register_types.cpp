#include "register_types.h"
#include "ipc_physics_server.h"
#include "ipc_body_state.h"

#include <gdextension_interface.h>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/defs.hpp>
#include <godot_cpp/core/memory.hpp>
#include <godot_cpp/godot.hpp>
#include <godot_cpp/classes/physics_server3d_manager.hpp>

using namespace godot;

static PhysicsServer3D *_create_physics_server() {
    return memnew(IPCPhysicsServer);
}

void initialize_m_abd_module(ModuleInitializationLevel p_level) {
    if (p_level == MODULE_INITIALIZATION_LEVEL_SERVERS) {
        // 클래스 등록
        ClassDB::register_class<IPCBodyState>();
        ClassDB::register_class<IPCPhysicsServer>();

        // 물리 서버 등록
        PhysicsServer3DManager::get_singleton()->register_server(
            "M-ABD",
            callable_mp_static(&_create_physics_server)
        );
    }
}

void uninitialize_m_abd_module(ModuleInitializationLevel p_level) {
    if (p_level == MODULE_INITIALIZATION_LEVEL_SERVERS) {
        // 서버는 Godot 엔진이 해제함
    }
}


extern "C" {

GDExtensionBool GDE_EXPORT m_abd_library_init(
    GDExtensionInterfaceGetProcAddress p_get_proc_address,
    const GDExtensionClassLibraryPtr p_library,
    GDExtensionInitialization *r_initialization)
{
    godot::GDExtensionBinding::InitObject init_obj(
        p_get_proc_address, p_library, r_initialization);

    init_obj.register_initializer(initialize_m_abd_module);
    init_obj.register_terminator(uninitialize_m_abd_module);
    init_obj.set_minimum_library_initialization_level(
        MODULE_INITIALIZATION_LEVEL_SERVERS);

    return init_obj.init();
}

} // extern "C"
